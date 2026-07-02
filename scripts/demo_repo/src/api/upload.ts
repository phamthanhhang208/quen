import { storage } from "./storage";

export async function uploadV2(file: Blob) {
  return storage.put(file);
}
