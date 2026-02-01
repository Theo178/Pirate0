"use client";

import { useState } from "react";
import { Loader2, Users, TrendingUp, TrendingDown, Minus } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { analyzeCharacterArc, type CharacterArc as CharacterArcType } from "@/lib/api";

export function CharacterArcAnalyzer() {
  const [sceneText, setSceneText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [arcs, setArcs] = useState<CharacterArcType[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!sceneText.trim()) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      const result = await analyzeCharacterArc(sceneText);
      setArcs(result);
    } catch (err) {
      setError("Failed to analyze character arc. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getSentimentIcon = (sentiment: number) => {
    if (sentiment > 2) return <TrendingUp className="w-4 h-4 text-foreground" />;
    if (sentiment < -2) return <TrendingDown className="w-4 h-4 text-muted-foreground" />;
    return <Minus className="w-4 h-4 text-muted-foreground" />;
  };

  const getIntensityBar = (intensity: number) => {
    return (
      <div className="flex gap-0.5">
        {Array.from({ length: 10 }).map((_, i) => (
          <div
            key={i}
            className={`w-2 h-4 rounded-sm ${
              i < intensity ? "bg-foreground" : "bg-border"
            }`}
          />
        ))}
      </div>
    );
  };

  const groupedArcs = arcs.reduce((acc, arc) => {
    if (!acc[arc.character]) acc[arc.character] = [];
    acc[arc.character].push(arc);
    return acc;
  }, {} as Record<string, CharacterArcType[]>);

  return (
    <div className="space-y-6">
      <Card className="border-border bg-card">
        <CardHeader className="pb-4">
          <CardTitle className="text-sm font-medium tracking-wide uppercase text-muted-foreground">
            Character Arc Analysis
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            placeholder="Paste your scene dialogue here..."
            value={sceneText}
            onChange={(e) => setSceneText(e.target.value)}
            className="min-h-[150px] font-mono text-sm bg-input border-border resize-none"
          />
          <Button
            onClick={handleAnalyze}
            disabled={isAnalyzing || !sceneText.trim()}
            className="w-full"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Analyzing Characters...
              </>
            ) : (
              <>
                <Users className="w-4 h-4 mr-2" />
                Analyze Character Arcs
              </>
            )}
          </Button>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </CardContent>
      </Card>

      {Object.keys(groupedArcs).length > 0 && (
        <div className="space-y-4">
          {Object.entries(groupedArcs).map(([character, beats]) => (
            <Card key={character} className="border-border bg-card">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-base font-mono">
                  {character}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {beats.map((beat, idx) => (
                    <div
                      key={idx}
                      className="p-4 bg-accent rounded-lg space-y-3"
                    >
                      <p className="text-sm font-mono italic">
                        {'"'}{beat.line}{'"'}
                      </p>
                      <div className="flex flex-wrap items-center gap-4 text-xs">
                        <div className="flex items-center gap-2">
                          <span className="text-muted-foreground uppercase tracking-wide">
                            Emotion
                          </span>
                          <span className="px-2 py-0.5 bg-secondary rounded">
                            {beat.emotion}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-muted-foreground uppercase tracking-wide">
                            Intensity
                          </span>
                          {getIntensityBar(beat.intensity)}
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-muted-foreground uppercase tracking-wide">
                            Sentiment
                          </span>
                          {getSentimentIcon(beat.sentiment)}
                          <span className="font-mono">{beat.sentiment}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
