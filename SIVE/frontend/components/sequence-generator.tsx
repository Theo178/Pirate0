"use client";

import { useState } from "react";
import {
  Loader2,
  Film,
  Eye,
  Camera,
  Users,
  Scissors,
  Clapperboard,
  Download
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { analyzeScene, generateSequence, type SceneAnalysis } from "@/lib/api";

export default function SequenceGenerator() {

  const [sceneText, setSceneText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<SceneAnalysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [downloadingId, setDownloadingId] = useState<number | null>(null);

  // -------------------------
  // Scene Analyzer Handler
  // -------------------------
  const handleAnalyze = async () => {

    if (!sceneText.trim()) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      const [analysisResult, shotsResult] = await Promise.all([
        analyzeScene(sceneText),
        generateSequence(sceneText)
      ]);

      setAnalysis({
        ...analysisResult,
        generated_shots: shotsResult
      });
    } catch (err) {
      setError("Failed to analyze scene. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // -------------------------
  // Download Handler (Proxy)
  // -------------------------
  const handleDownload = async (url: string, id: number) => {

    try {
      setDownloadingId(id);

      const proxyUrl = `/api/image-proxy?url=${encodeURIComponent(url)}`;

      const link = document.createElement("a");
      link.href = proxyUrl;
      link.download = `shot_${id}.png`;

      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

    } finally {
      setDownloadingId(null);
    }
  };

  return (
    <div className="space-y-6">

      {/* =============================
          Scene Input Section
      ============================== */}

      <Card className="border-border bg-card">

        <CardHeader className="pb-4">
          <CardTitle className="text-sm font-medium tracking-wide uppercase text-muted-foreground">
            Scene Text
          </CardTitle>
        </CardHeader>

        <CardContent className="space-y-4">

          <Textarea
            placeholder={`EXT. ALLEYWAY - NIGHT

A lone figure stands beneath the flickering neon sign...`}
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

          {error && (
            <p className="text-sm text-destructive">{error}</p>
          )}

        </CardContent>

      </Card>

      {/* =============================
          Analysis Output
      ============================== */}

      {analysis && (

        <div className="grid gap-4 md:grid-cols-2">

          {/* Scene Intent */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <Eye className="w-4 h-4" />
                Scene Intent
              </CardTitle>
            </CardHeader>

            <CardContent>
              <p><b>Emotion:</b> {analysis.scene_intent.emotion}</p>
              <p><b>Story Purpose:</b> {analysis.scene_intent.story_purpose}</p>
            </CardContent>
          </Card>

          {/* Visual Mood */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <Clapperboard className="w-4 h-4" />
                Visual Mood
              </CardTitle>
            </CardHeader>

            <CardContent>
              <p><b>Lighting:</b> {analysis.visual_mood.lighting_style}</p>
              <p><b>Color Palette:</b> {analysis.visual_mood.color_palette}</p>
            </CardContent>
          </Card>

          {/* Camera Language */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <Camera className="w-4 h-4" />
                Camera Language
              </CardTitle>
            </CardHeader>

            <CardContent className="space-y-2">
              <p><b>Motion:</b> {analysis.camera_language.camera_motion}</p>

              {analysis.camera_language.shot_plan.map((shot, idx) => (
                <div key={idx} className="text-xs bg-secondary/20 p-2 rounded">
                  {shot.shot} — {shot.reason}
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Actor Blocking */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <Users className="w-4 h-4" />
                Actor Blocking
              </CardTitle>
            </CardHeader>

            <CardContent className="space-y-2">
              {analysis.actor_blocking.map((block, idx) => (
                <div key={idx} className="p-2 border rounded text-xs">
                  <b>{block.character}</b> — {block.position}
                  <div>{block.movement}</div>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Editing Rhythm */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <Scissors className="w-4 h-4" />
                Editing Rhythm
              </CardTitle>
            </CardHeader>

            <CardContent>
              {Object.entries(analysis.editing_rhythm).map(([k, v]) => (
                <p key={k}><b>{k}:</b> {v}</p>
              ))}
            </CardContent>
          </Card>

        </div>
      )}

      {/* =============================
          Generated Storyboard Frames
      ============================== */}

      {analysis?.generated_shots && (

        <Card className="border-border bg-card">

          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm">
              <Film className="w-4 h-4" />
              Generated Shot Sequence
            </CardTitle>
          </CardHeader>

          <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-4">

            {analysis.generated_shots.map((shot) => (

              <div
                key={shot.shot_id}
                className="p-3 rounded-lg border bg-secondary/10 space-y-2"
              >

                <div className="text-xs font-semibold">
                  {shot.type} Shot
                </div>

                <img
                  src={`/api/image-proxy?url=${encodeURIComponent(shot.url)}`}
                  className="rounded w-full aspect-video object-cover border"
                  loading="lazy"
                />

                <p className="text-xs text-muted-foreground">
                  {shot.prompt}
                </p>

                <Button
                  size="sm"
                  variant="secondary"
                  className="w-full"
                  onClick={() => handleDownload(shot.url, shot.shot_id)}
                  disabled={downloadingId === shot.shot_id}
                >
                  {downloadingId === shot.shot_id ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Downloading...
                    </>
                  ) : (
                    <>
                      <Download className="w-4 h-4 mr-2" />
                      Download Frame
                    </>
                  )}
                </Button>

              </div>

            ))}

          </CardContent>

        </Card>
      )}

    </div>
  );
}