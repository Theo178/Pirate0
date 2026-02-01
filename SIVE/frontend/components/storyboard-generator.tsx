"use client";

import { useState } from "react";
import { Loader2, ImageIcon, Sparkles, Copy, Check } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { generateStoryboard, type StoryboardImage } from "@/lib/api";

export function StoryboardGenerator() {
  const [sceneText, setSceneText] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [storyboard, setStoryboard] = useState<StoryboardImage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const handleGenerate = async () => {
    if (!sceneText.trim()) return;

    setIsGenerating(true);
    setError(null);

    try {
      const result = await generateStoryboard(sceneText);
      setStoryboard(result);
    } catch (err) {
      setError("Failed to generate storyboard. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  };

  const copyPrompt = async () => {
    if (storyboard?.image_prompt) {
      await navigator.clipboard.writeText(storyboard.image_prompt);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="space-y-6">
      <Card className="border-border bg-card">
        <CardHeader className="pb-4">
          <CardTitle className="text-sm font-medium tracking-wide uppercase text-muted-foreground">
            Generate Storyboard
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            placeholder="Describe your scene for the iconic shot..."
            value={sceneText}
            onChange={(e) => setSceneText(e.target.value)}
            className="min-h-[150px] font-mono text-sm bg-input border-border resize-none"
          />
          <Button
            onClick={handleGenerate}
            disabled={isGenerating || !sceneText.trim()}
            className="w-full"
          >
            {isGenerating ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Generating Storyboard...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4 mr-2" />
                Generate Iconic Shot
              </>
            )}
          </Button>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </CardContent>
      </Card>

      {storyboard && (
        <Card className="border-border bg-card overflow-hidden">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center justify-between text-sm font-medium">
              <span className="flex items-center gap-2">
                <ImageIcon className="w-4 h-4" />
                Generated Storyboard
              </span>
              <Button
                variant="ghost"
                size="sm"
                onClick={copyPrompt}
                className="text-xs"
              >
                {copied ? (
                  <>
                    <Check className="w-3 h-3 mr-1" />
                    Copied
                  </>
                ) : (
                  <>
                    <Copy className="w-3 h-3 mr-1" />
                    Copy Prompt
                  </>
                )}
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="relative aspect-video bg-accent rounded-lg overflow-hidden">
              <img
                src={storyboard.image_url || "/placeholder.svg"}
                alt="Generated storyboard"
                className="w-full h-full object-cover"
                crossOrigin="anonymous"
              />
            </div>
            <div className="p-3 bg-accent rounded-lg">
              <p className="text-xs text-muted-foreground uppercase tracking-wide mb-2">
                Image Prompt
              </p>
              <p className="text-sm font-mono text-muted-foreground">
                {storyboard.image_prompt}
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
