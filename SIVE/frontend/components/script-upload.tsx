"use client";

import React from "react"

import { useState, useCallback } from "react";
import { Upload, FileText, X, Loader2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { uploadScript, type ContextAnalysis } from "@/lib/api";

interface ScriptUploadProps {
  onUploadComplete: (analysis: ContextAnalysis, chunksCount: number) => void;
}

export function ScriptUpload({ onUploadComplete }: ScriptUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && isValidFile(droppedFile)) {
      setFile(droppedFile);
      setError(null);
    } else {
      setError("Please upload a .pdf, .txt, or .md file");
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile && isValidFile(selectedFile)) {
      setFile(selectedFile);
      setError(null);
    } else {
      setError("Please upload a .pdf, .txt, or .md file");
    }
  }, []);

  const isValidFile = (file: File) => {
    const validTypes = [".pdf", ".txt", ".md"];
    return validTypes.some((type) => file.name.toLowerCase().endsWith(type));
  };

  const handleUpload = async () => {
    if (!file) return;

    setIsUploading(true);
    setError(null);

    try {
      const response = await uploadScript(file);
      onUploadComplete(response.context_analysis, response.chunks_count);
    } catch (err) {
      setError("Failed to upload script. Please try again.");
    } finally {
      setIsUploading(false);
    }
  };

  const removeFile = () => {
    setFile(null);
    setError(null);
  };

  return (
    <Card className="border-border bg-card">
      <CardHeader className="pb-4">
        <CardTitle className="text-sm font-medium tracking-wide uppercase text-muted-foreground">
          Upload Script
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`
            relative border-2 border-dashed rounded-lg p-8 text-center transition-all cursor-pointer
            ${isDragging ? "border-foreground bg-accent" : "border-border hover:border-muted-foreground"}
          `}
        >
          <input
            type="file"
            accept=".pdf,.txt,.md"
            onChange={handleFileSelect}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
          {file ? (
            <div className="flex items-center justify-center gap-3">
              <FileText className="w-5 h-5 text-foreground" />
              <span className="text-sm font-mono">{file.name}</span>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  removeFile();
                }}
                className="p-1 hover:bg-accent rounded"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <div className="space-y-2">
              <Upload className="w-8 h-8 mx-auto text-muted-foreground" />
              <p className="text-sm text-muted-foreground">
                Drop your script here or click to browse
              </p>
              <p className="text-xs text-muted-foreground/60">
                Supports PDF, TXT, MD
              </p>
            </div>
          )}
        </div>

        {error && (
          <p className="mt-3 text-sm text-destructive">{error}</p>
        )}

        {file && (
          <Button
            onClick={handleUpload}
            disabled={isUploading}
            className="w-full mt-4"
          >
            {isUploading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Processing Script...
              </>
            ) : (
              "Initialize Session"
            )}
          </Button>
        )}
      </CardContent>
    </Card>
  );
}
