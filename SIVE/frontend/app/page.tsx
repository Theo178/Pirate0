"use client";

import { useState } from "react";
import { Film, Users, ImageIcon, Layers, FileText, Menu, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { ScriptUpload } from "@/components/script-upload";
import { SceneAnalyzer } from "@/components/scene-analyzer";
import { CharacterArcAnalyzer } from "@/components/character-arc";
import { StoryboardGenerator } from "@/components/storyboard-generator";
import SequenceGenerator from "@/components/sequence-generator";
import { ContextPanel } from "@/components/context-panel";
import type { ContextAnalysis, SceneAnalysis } from "@/lib/api";

type Tab = "upload" | "analyze" | "characters" | "storyboard" | "sequence";

const tabs = [
  { id: "upload" as Tab, label: "Upload", icon: FileText },
  { id: "analyze" as Tab, label: "Scene Analysis", icon: Film },
  { id: "characters" as Tab, label: "Character Arcs", icon: Users },
  { id: "storyboard" as Tab, label: "Storyboard", icon: ImageIcon },
  { id: "sequence" as Tab, label: "Sequence", icon: Layers },
];

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<Tab>("upload");
  const [contextAnalysis, setContextAnalysis] = useState<ContextAnalysis | null>(null);
  const [chunksCount, setChunksCount] = useState(0);
  const [sceneAnalysis, setSceneAnalysis] = useState<SceneAnalysis | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const handleUploadComplete = (analysis: ContextAnalysis, chunks: number) => {
    setContextAnalysis(analysis);
    setChunksCount(chunks);
    setActiveTab("analyze");
  };

  const handleSceneAnalysisComplete = (analysis: SceneAnalysis) => {
    setSceneAnalysis(analysis);
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="flex h-14 items-center justify-between px-4 lg:px-6">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="sm"
              className="lg:hidden"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              {sidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </Button>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-foreground rounded flex items-center justify-center">
                <Film className="w-4 h-4 text-background" />
              </div>
              <span className="text-sm font-medium tracking-tight">SIVE</span>
              <span className="hidden sm:inline text-xs text-muted-foreground font-mono">
                Scene Intent & Visual Engine
              </span>
            </div>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar */}
        <aside
          className={cn(
            "fixed inset-y-0 left-0 z-40 w-64 border-r border-border bg-sidebar pt-14 transition-transform lg:static lg:translate-x-0",
            sidebarOpen ? "translate-x-0" : "-translate-x-full"
          )}
        >
          <nav className="flex flex-col gap-1 p-4">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => {
                    setActiveTab(tab.id);
                    setSidebarOpen(false);
                  }}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2.5 rounded-md text-sm transition-colors",
                    activeTab === tab.id
                      ? "bg-sidebar-accent text-sidebar-foreground"
                      : "text-muted-foreground hover:text-sidebar-foreground hover:bg-sidebar-accent/50"
                  )}
                >
                  <Icon className="w-4 h-4" />
                  {tab.label}
                </button>
              );
            })}
          </nav>

          {contextAnalysis && (
            <div className="p-4 border-t border-sidebar-border">
              <ContextPanel analysis={contextAnalysis} chunksCount={chunksCount} />
            </div>
          )}
        </aside>

        {/* Overlay for mobile */}
        {sidebarOpen && (
          <button
            type="button"
            className="fixed inset-0 z-30 bg-background/80 backdrop-blur-sm lg:hidden"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close sidebar"
          />
        )}

        {/* Main Content */}
        <main className="flex-1 p-4 lg:p-8 min-h-[calc(100vh-3.5rem)]">
          <div className="max-w-5xl mx-auto">
            {/* Page Header */}
            <div className="mb-8">
              <h1 className="text-2xl font-medium tracking-tight">
                {tabs.find((t) => t.id === activeTab)?.label}
              </h1>
              <p className="text-sm text-muted-foreground mt-1">
                {activeTab === "upload" && "Upload your script to initialize the session and extract context."}
                {activeTab === "analyze" && "Perform deep cinematic analysis of your scenes."}
                {activeTab === "characters" && "Analyze emotional beats and sentiment of characters."}
                {activeTab === "storyboard" && "Generate AI-powered storyboard frames."}
                {activeTab === "sequence" && "Create a complete shot sequence for your scene."}
              </p>
            </div>

            {/* Tab Content */}
            <div className="space-y-6">
              {activeTab === "upload" && (
                <ScriptUpload onUploadComplete={handleUploadComplete} />
              )}
              {activeTab === "analyze" && (
                <SceneAnalyzer onAnalysisComplete={handleSceneAnalysisComplete} />
              )}
              {activeTab === "characters" && <CharacterArcAnalyzer />}
              {activeTab === "storyboard" && <StoryboardGenerator />}
              {activeTab === "sequence" && <SequenceGenerator />}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
