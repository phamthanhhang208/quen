#!/usr/bin/env python3
"""Provision a Quên demo box on Alibaba Cloud ECS — one command, idempotent.

    ALIBABA_CLOUD_ACCESS_KEY_ID=... ALIBABA_CLOUD_ACCESS_KEY_SECRET=... \
      .venv/bin/python deploy/provision.py [--region ap-southeast-1] [--enable-live]

Reads the keys from the environment or .env (never prints them). Creates —
or reuses, by name — a security group (22/80 open) and one `quen-demo`
instance whose cloud-init clones this repo's branch and runs
deploy/setup.sh in OFFLINE demo mode. The DashScope key deliberately never
enters user_data (instance metadata is readable by any process on the box);
to switch the box to live mode afterwards run

    .venv/bin/python deploy/provision.py --enable-live

which delivers the key via Cloud Assistant RunCommand instead — or paste
the one-liner it prints into the console Workbench yourself.

Needs RAM permissions: AliyunECSFullAccess + AliyunVPCFullAccess.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.request

from dotenv import load_dotenv

REPO = "https://github.com/phamthanhhang208/quen.git"
BRANCH = "claude/quen-memory-agent-spec-p3izss"
NAME = "quen-demo"
INSTANCE_TYPES = [  # tried in order until one has stock
    "ecs.e-c1m1.large", "ecs.e-c1m2.large", "ecs.t6-c1m2.large",
    "ecs.u1-c1m1.large", "ecs.n4.large",
]


def clients(region: str):
    from alibabacloud_ecs20140526.client import Client as Ecs
    from alibabacloud_tea_openapi import models as oapi
    from alibabacloud_vpc20160428.client import Client as Vpc

    kid = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID")
    sec = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET")
    if not (kid and sec):
        raise SystemExit(
            "set ALIBABA_CLOUD_ACCESS_KEY_ID / ALIBABA_CLOUD_ACCESS_KEY_SECRET "
            "(env or .env)"
        )

    def cfg(endpoint):
        return oapi.Config(access_key_id=kid, access_key_secret=sec,
                           region_id=region, endpoint=endpoint)

    return (Ecs(cfg(f"ecs.{region}.aliyuncs.com")),
            Vpc(cfg(f"vpc.{region}.aliyuncs.com")))


def ensure_network(ecs, vpc, region: str) -> tuple[str, str, str]:
    """-> (vpc_id, vswitch_id, zone_id): reuse anything present, else create
    the account's default VPC."""
    from alibabacloud_vpc20160428 import models as vm

    def list_vswitches():
        return vpc.describe_vswitches(
            vm.DescribeVSwitchesRequest(region_id=region, page_size=50)
        ).body.v_switches.v_switch

    vsws = list_vswitches()
    if not vsws:
        vpcs = vpc.describe_vpcs(
            vm.DescribeVpcsRequest(region_id=region)
        ).body.vpcs.vpc
        if not vpcs:
            print("==> creating the default VPC")
            vpc.create_default_vpc(vm.CreateDefaultVpcRequest(region_id=region))
            time.sleep(5)
        # CreateDefaultVpc does NOT create VSwitches (the console does both,
        # the API does not) — create one per the first zone that accepts it
        zones = vpc.describe_zones(
            vm.DescribeZonesRequest(region_id=region)
        ).body.zones.zone
        for z in zones:
            try:
                print(f"==> creating default VSwitch in {z.zone_id}")
                vpc.create_default_vswitch(vm.CreateDefaultVSwitchRequest(
                    region_id=region, zone_id=z.zone_id))
                break
            except Exception as exc:
                print(f"    {z.zone_id}: {str(exc)[:90]}")
        for _ in range(30):
            time.sleep(4)
            vsws = list_vswitches()
            if vsws:
                break
        else:
            raise SystemExit("could not obtain a VSwitch in any zone")
    sw = vsws[0]
    return sw.vpc_id, sw.v_switch_id, sw.zone_id


def ensure_security_group(ecs, region: str, vpc_id: str) -> str:
    from alibabacloud_ecs20140526 import models as em

    got = ecs.describe_security_groups(em.DescribeSecurityGroupsRequest(
        region_id=region, security_group_name=f"{NAME}-sg", vpc_id=vpc_id,
    )).body.security_groups.security_group
    if got:
        return got[0].security_group_id
    sg = ecs.create_security_group(em.CreateSecurityGroupRequest(
        region_id=region, security_group_name=f"{NAME}-sg", vpc_id=vpc_id,
        description="quen demo: ssh + http",
    )).body.security_group_id
    for port in ("22/22", "80/80"):
        ecs.authorize_security_group(em.AuthorizeSecurityGroupRequest(
            region_id=region, security_group_id=sg, ip_protocol="tcp",
            port_range=port, source_cidr_ip="0.0.0.0/0",
        ))
    print(f"==> security group {sg} (22, 80 open)")
    return sg


def pick_image(ecs, region: str) -> tuple[str, int]:
    from alibabacloud_ecs20140526 import models as em

    imgs = ecs.describe_images(em.DescribeImagesRequest(
        region_id=region, ostype="linux", architecture="x86_64",
        image_owner_alias="system", image_name="ubuntu_24_04_x64*",
        page_size=10, status="Available",
    )).body.images.image
    if not imgs:
        raise SystemExit("no ubuntu_24_04_x64 system image in this region")
    img = min(imgs, key=lambda i: i.size)  # base image, not the GPU builds
    print(f"==> image {img.image_id} ({img.size} GB)")
    return img.image_id, img.size


def existing_instance(ecs, region: str):
    from alibabacloud_ecs20140526 import models as em

    got = ecs.describe_instances(em.DescribeInstancesRequest(
        region_id=region, instance_name=NAME,
    )).body.instances.instance
    return got[0] if got else None


def user_data() -> str:
    # no secrets in here, ever: user_data is world-readable on the instance
    script = f"""#!/bin/bash
exec > /var/log/quen-bootstrap.log 2>&1
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get install -y git
git clone -b {BRANCH} {REPO} /opt/quen
bash /opt/quen/deploy/setup.sh
"""
    return base64.b64encode(script.encode()).decode()


def run_instance(ecs, region, image, image_gb, sg, vsw, zone, udata) -> str:
    from alibabacloud_ecs20140526 import models as em

    last = None
    for itype in INSTANCE_TYPES:
        for category in ("cloud_essd", "cloud_efficiency"):
            try:
                ids = ecs.run_instances(em.RunInstancesRequest(
                    region_id=region, image_id=image, instance_type=itype,
                    security_group_id=sg, v_switch_id=vsw, zone_id=zone,
                    instance_name=NAME, host_name=NAME,
                    instance_charge_type="PostPaid",
                    internet_charge_type="PayByTraffic",
                    internet_max_bandwidth_out=5,
                    system_disk=em.RunInstancesRequestSystemDisk(
                        size=str(max(40, image_gb)), category=category),
                    user_data=udata,
                    tag=[em.RunInstancesRequestTag(key="app", value="quen")],
                )).body.instance_id_sets.instance_id_set
                print(f"==> launched {ids[0]} ({itype}, {category})")
                return ids[0]
            except Exception as exc:  # no stock / unsupported combo → next
                last = exc
                msg = getattr(exc, "message", str(exc))
                print(f"    {itype}/{category}: {msg[:100]}")
                if "NotSupportDiskCategory" not in str(exc):
                    break  # different cause — other categories won't help
    raise SystemExit(f"no instance type available: {last}")


def wait_public_ip(ecs, region: str, iid: str) -> str:
    from alibabacloud_ecs20140526 import models as em

    for _ in range(60):
        inst = ecs.describe_instances(em.DescribeInstancesRequest(
            region_id=region, instance_ids=json.dumps([iid]),
        )).body.instances.instance[0]
        ips = inst.public_ip_address.ip_address
        if inst.status == "Running" and ips:
            return ips[0]
        time.sleep(5)
    raise SystemExit("instance did not reach Running with a public IP")


def poll_health(ip: str, tries: int = 100) -> bool:
    url = f"http://{ip}/vitals"
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        if i % 10 == 9:
            print(f"    still bootstrapping ({(i + 1) * 10}s)…")
        time.sleep(10)
    return False


LIVE_SWITCH = (
    "printf 'DASHSCOPE_API_KEY=%s\\n' \"$KEY\" > /opt/quen/.env && "
    "chmod 600 /opt/quen/.env && bash /opt/quen/deploy/setup.sh"
)


def enable_live(ecs, region: str) -> None:
    """Deliver the DashScope key via Cloud Assistant (not user_data) and
    re-run setup so the systemd unit flips from OFFLINE to live."""
    from alibabacloud_ecs20140526 import models as em

    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("--enable-live needs DASHSCOPE_API_KEY in env/.env")
    inst = existing_instance(ecs, region)
    if not inst or inst.status != "Running":
        raise SystemExit("no Running quen-demo instance to switch")
    content = f'KEY="{key}"\n{LIVE_SWITCH}\n'
    resp = ecs.run_command(em.RunCommandRequest(
        region_id=region, type="RunShellScript", name="quen-enable-live",
        command_content=content, instance_id=[inst.instance_id], timeout=600,
    ))
    print(f"==> live-mode switch sent (invocation "
          f"{resp.body.invoke_id}); the service restarts in ~1 min.")
    print("    (manual alternative — paste in console Workbench:")
    print(f"     KEY=sk-...; {LIVE_SWITCH})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="ap-southeast-1")
    ap.add_argument("--enable-live", action="store_true",
                    help="switch an existing quen-demo box to live Qwen mode")
    args = ap.parse_args()

    load_dotenv()
    ecs, vpc = clients(args.region)

    if args.enable_live:
        enable_live(ecs, args.region)
        return

    inst = existing_instance(ecs, args.region)
    if inst:
        ip = (inst.public_ip_address.ip_address or ["<no public ip>"])[0]
        print(f"==> instance already exists: {inst.instance_id} "
              f"({inst.status}) http://{ip}/")
        sys.exit(0)

    vpc_id, vsw, zone = ensure_network(ecs, vpc, args.region)
    sg = ensure_security_group(ecs, args.region, vpc_id)
    image, image_gb = pick_image(ecs, args.region)
    print("==> boot mode: OFFLINE demo (live switch is a separate step)")

    iid = run_instance(ecs, args.region, image, image_gb, sg, vsw, zone,
                       user_data())
    ip = wait_public_ip(ecs, args.region, iid)
    print(f"==> running: {iid} — bootstrapping Quên (apt + build, ~5-8 min)")
    ok = poll_health(ip)
    print()
    state = "UP" if ok else (
        "not answering yet — check again in a few minutes; on the box: "
        "tail -f /var/log/quen-bootstrap.log"
    )
    print(f"    dashboard : http://{ip}/")
    print(f"    health    : http://{ip}/vitals  ({state})")


if __name__ == "__main__":
    main()
