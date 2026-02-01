"use client";

import { useState } from "react";
import { Loader2, Film, Eye, Camera, Users, Scissors, Clapperboard } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { analyzeScene, type SceneAnalysis } from "@/lib/api";

interface SceneAnalyzerProps {
  onAnalysisComplete: (analysis: SceneAnalysis) => void;
}

export function SceneAnalyzer({ onAnalysisComplete }: SceneAnalyzerProps) {
  const [sceneText, setSceneText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<SceneAnalysis | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!sceneText.trim()) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      const result = await analyzeScene(sceneText);
      setAnalysis(result);
      onAnalysisComplete(result);
    } catch (err) {
      setError("Failed to analyze scene. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card className="border-border bg-card">
        <CardHeader className="pb-4">
          <CardTitle className="text-sm font-medium tracking-wide uppercase text-muted-foreground">
            Scene Text
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            placeholder="EXT. ALLEYWAY - NIGHT

A lone figure stands beneath the flickering neon sign..."
            value={sceneText}
            onChange={(e) => setSceneText(e.target.value)}
            className="min-h-[200px] font-mono text-sm bg-input border-border resize-none"
          />
          <Button
            onClick={handleAnalyze}
            disabled={isAnalyzing || !sceneText.trim()}
            className="w-full"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Analyzing Scene...
              </>
            ) : (
              <>
                <Film className="w-4 h-4 mr-2" />
                Analyze Scene
              </>
            )}
          </Button>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </CardContent>
      </Card>

      {analysis && (
        <div className="grid gap-4 md:grid-cols-2">
          <Card className="border-border bg-card">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-sm font-medium">
                <Eye className="w-4 h-4" />
                Scene Intent
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wide">Emotion</p>
                <p className="text-sm mt-1">{analysis.scene_intent.emotion}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wide">Story Purpose</p>
                <p className="text-sm mt-1">{analysis.scene_intent.story_purpose}</p>
              </div>
            </CardContent>
          </Card>

          <Card className="border-border bg-card">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-sm font-medium">
                <Clapperboard className="w-4 h-4" />
                Visual Mood
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wide">Lighting</p>
                <p className="text-sm mt-1">{analysis.visual_mood.lighting_style}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wide">Color Palette</p>
                <p className="text-sm mt-1">{analysis.visual_mood.color_palette}</p>
              </div>
            </CardContent>
          </Card>

          <Card className="border-border bg-card">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-sm font-medium">
                <Camera className="w-4 h-4" />
                Camera Language
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wide">Motion</p>
                <p className="text-sm mt-1">{analysis.camera_language.camera_motion}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wide">Shot Plan</p>
                <div className="mt-2 space-y-2">
                  {analysis.camera_language.shot_plan.map((shot, idx) => (
                    <div key={idx} className="p-2 bg-accent rounded text-xs">
                      <span className="font-medium">{shot.shot}</span>
                      <span className="text-muted-foreground"> — {shot.reason}</span>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-border bg-card">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-sm font-medium">
                <Users className="w-4 h-4" />
                Actor Blocking
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {analysis.actor_blocking.map((block, idx) => (
                  <div key={idx} className="p-3 bg-secondary/20 rounded-lg space-y-2 border border-border/50">
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-sm">{block.character}</span>
                      <span className="text-xs text-muted-foreground bg-background/50 px-2 py-0.5 rounded">
                        {block.position}
                      </span>
                    </div>
                    <div className="space-y-1.5 pt-1">
                      <div>
                        <span className="text-[10px] uppercase tracking-wider text-muted-foreground/70">Movement</span>
                        <p className="text-sm text-muted-foreground leading-snug">{block.movement}</p>
                      </div>
                      <div className="grid grid-cols-2 gap-2 pt-1">
                        <div>
                          <span className="text-[10px] uppercase tracking-wider text-muted-foreground/70">Posture</span>
                          <p className="text-xs text-muted-foreground">{block.posture}</p>
                        </div>
                        <div>
                          <span className="text-[10px] uppercase tracking-wider text-muted-foreground/70">Eye Focus</span>
                          <p className="text-xs text-muted-foreground">{block.eye_focus}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="border-border bg-card">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-sm font-medium">
                <Scissors className="w-4 h-4" />
                Editing Rhythm
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {Object.entries(analysis.editing_rhythm).map(([key, value]) => (
                  <div key={key}>
                    <p className="text-xs text-muted-foreground uppercase tracking-wide">{key}</p>
                    <p className="text-sm mt-1">{value}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="border-border bg-card">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-sm font-medium">
                <Film className="w-4 h-4" />
                Production Logistics
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {Object.entries(analysis.production_logistics).map(([key, value]) => (
                  <div key={key}>
                    <p className="text-xs text-muted-foreground uppercase tracking-wide">{key}</p>
                    <p className="text-sm mt-1">{value}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
