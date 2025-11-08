import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Sparkles, ArrowRight } from "lucide-react";

interface ClassificationResult {
  label: string;
  confidence: number;
}

const mockClassify = (text: string): ClassificationResult[] => {
  // Simple mock classification based on keywords
  const results: ClassificationResult[] = [];
  const lowerText = text.toLowerCase();
  
  if (lowerText.includes("happy") || lowerText.includes("joy") || lowerText.includes("great")) {
    results.push({ label: "Positive", confidence: 0.92 });
    results.push({ label: "Neutral", confidence: 0.06 });
    results.push({ label: "Negative", confidence: 0.02 });
  } else if (lowerText.includes("sad") || lowerText.includes("bad") || lowerText.includes("terrible")) {
    results.push({ label: "Negative", confidence: 0.88 });
    results.push({ label: "Neutral", confidence: 0.09 });
    results.push({ label: "Positive", confidence: 0.03 });
  } else if (lowerText.includes("tech") || lowerText.includes("science") || lowerText.includes("computer")) {
    results.push({ label: "Technology", confidence: 0.85 });
    results.push({ label: "Science", confidence: 0.12 });
    results.push({ label: "Business", confidence: 0.03 });
  } else if (lowerText.includes("sports") || lowerText.includes("game") || lowerText.includes("play")) {
    results.push({ label: "Sports", confidence: 0.91 });
    results.push({ label: "Entertainment", confidence: 0.07 });
    results.push({ label: "News", confidence: 0.02 });
  } else {
    results.push({ label: "Neutral", confidence: 0.75 });
    results.push({ label: "General", confidence: 0.15 });
    results.push({ label: "Other", confidence: 0.10 });
  }
  
  return results;
};

export const ClassificationDemo = () => {
  const [inputText, setInputText] = useState("");
  const [results, setResults] = useState<ClassificationResult[] | null>(null);
  const [isClassifying, setIsClassifying] = useState(false);

  const handleClassify = () => {
    if (!inputText.trim()) return;
    
    setIsClassifying(true);
    
    // Simulate processing time (3000ms = 3 seconds)
    setTimeout(() => {
      const classificationResults = mockClassify(inputText);
      setResults(classificationResults);
      setIsClassifying(false);
    }, 3000);
  };

  return (
    <div className="w-full max-w-4xl mx-auto space-y-8">
      <Card className="p-8 shadow-soft">
        <div className="flex flex-col items-center justify-center space-y-6 min-h-[300px]">
          {/* Circular Processing Indicator */}
          <div className="relative">
            <div className="w-32 h-32 rounded-full border-4 border-muted flex items-center justify-center">
              {isClassifying ? (
                <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-primary animate-spin" />
              ) : null}
              <Sparkles className={`w-12 h-12 ${isClassifying ? 'text-primary animate-pulse' : 'text-muted-foreground'}`} />
            </div>
          </div>
          
          <div className="text-center space-y-2">
            <h3 className="text-lg font-semibold text-foreground">
              {isClassifying ? "Processing Input from ESP32..." : "Waiting for ESP32 Data"}
            </h3>
            <p className="text-sm text-muted-foreground">
              {isClassifying ? "Model is analyzing the input" : "Ready to receive and classify"}
            </p>
          </div>

          {/* Hidden textarea for testing */}
          <div className="w-full space-y-2">
            <Textarea
              placeholder="Simulate ESP32 input (for testing)..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              className="min-h-[80px] resize-none"
            />
            <Button
              onClick={handleClassify}
              disabled={!inputText.trim() || isClassifying}
              className="w-full"
              variant="hero"
            >
              Simulate Processing
            </Button>
          </div>
        </div>
      </Card>

      {results && (
        <Card className="p-8 shadow-soft animate-in fade-in slide-in-from-bottom-4 duration-500">
          <div className="space-y-6">
            <div className="flex items-center gap-2">
              <Sparkles className="text-primary" />
              <h3 className="text-xl font-semibold text-foreground">Classification Results</h3>
            </div>
            
            <div className="space-y-4">
              {results.map((result, index) => (
                <div key={index} className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-foreground">{result.label}</span>
                    <span className="text-sm text-muted-foreground">
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="relative h-3 bg-muted rounded-full overflow-hidden">
                    <div
                      className="absolute inset-y-0 left-0 bg-gradient-primary rounded-full transition-all duration-700 ease-out"
                      style={{ width: `${result.confidence * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};
