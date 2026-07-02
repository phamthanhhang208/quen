import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { MemoriesTable } from "@/panels/MemoriesTable";
import { MemoryDetail } from "@/panels/MemoryDetail";
import { DreamLog } from "@/panels/DreamLog";
import { RecallTrace } from "@/panels/RecallTrace";
import { Vitals } from "@/panels/Vitals";

export default function App() {
  const [selectedId, setSelectedId] = useState<string | null>(null);

  return (
    <div className="mx-auto max-w-[1400px] px-4 py-4">
      <header className="mb-4 flex items-baseline gap-3">
        <h1 className="text-xl font-semibold tracking-tight">Quên · memory</h1>
        <span className="text-sm text-zinc-500">
          trust-calibrated forgetting
        </span>
      </header>

      <Tabs defaultValue="memories">
        <TabsList>
          <TabsTrigger value="memories">Memories</TabsTrigger>
          <TabsTrigger value="dream">Dream log</TabsTrigger>
          <TabsTrigger value="trace">Recall trace</TabsTrigger>
          <TabsTrigger value="vitals">Vitals</TabsTrigger>
        </TabsList>

        <TabsContent value="memories">
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
            <MemoriesTable selectedId={selectedId} onSelect={setSelectedId} />
            {selectedId ? (
              <MemoryDetail id={selectedId} onSelect={setSelectedId} />
            ) : (
              <div className="hidden items-center justify-center rounded-xl border border-dashed border-zinc-200 text-sm text-zinc-400 xl:flex">
                select a memory to inspect it
              </div>
            )}
          </div>
        </TabsContent>

        <TabsContent value="dream">
          <DreamLog />
        </TabsContent>

        <TabsContent value="trace">
          <RecallTrace onSelectMemory={setSelectedId} />
        </TabsContent>

        <TabsContent value="vitals">
          <Vitals />
        </TabsContent>
      </Tabs>
    </div>
  );
}
