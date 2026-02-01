"use client";

import { Users, Film, Palette } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ContextAnalysis } from "@/lib/api";

interface ContextPanelProps {
  analysis: ContextAnalysis;
  chunksCount: number;
}

export function ContextPanel({ analysis, chunksCount }: ContextPanelProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium tracking-wide uppercase text-muted-foreground">
          Script Context
        </h3>
        <span className="text-xs text-muted-foreground font-mono">
          {chunksCount} chunks indexed
        </span>
      </div>

      <Card className="border-border bg-card">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-sm font-medium">
            <Film className="w-4 h-4" />
            Genre
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-lg font-medium">{analysis.genre}</p>
        </CardContent>
      </Card>

      <Card className="border-border bg-card">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-sm font-medium">
            <Palette className="w-4 h-4" />
            Visual Style
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground leading-relaxed">
            {analysis.visual_style}
          </p>
        </CardContent>
      </Card>

      <Card className="border-border bg-card">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-sm font-medium">
            <Users className="w-4 h-4" />
            Characters
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {analysis.characters.map((character, idx) => {
              // Handle both string format "Name: description" and object format { name, description }
              let name: string;
              let description: string | undefined;
              
              if (typeof character === "string") {
                const [namePart, ...descParts] = character.split(":");
                name = namePart;
                description = descParts.join(":").trim() || undefined;
              } else if (typeof character === "object" && character !== null) {
                const charObj = character as { name?: string; description?: string };
                name = charObj.name || "Unknown";
                description = charObj.description;
              } else {
                name = String(character);
                description = undefined;
              }
              
              return (
                <div key={idx} className="p-3 bg-accent rounded-lg">
                  <p className="text-sm font-medium font-mono">{name}</p>
                  {description && (
                    <p className="text-xs text-muted-foreground mt-1">
                      {description}
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
