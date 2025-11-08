import { ClassificationDemo } from "@/components/ClassificationDemo";
import { Sparkles, Zap, Shield } from "lucide-react";

const Index = () => {
  return (
    <div className="min-h-screen bg-gradient-hero">
      {/* Hero Section */}
      <section className="container mx-auto px-4 pt-20 pb-16">
        <div className="text-center space-y-6 max-w-3xl mx-auto">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-card border border-border rounded-full text-sm">
            <Sparkles className="w-4 h-4 text-primary" />
            <span className="text-muted-foreground">AI-Powered Text Classification</span>
          </div>
          
          <h1 className="text-5xl md:text-6xl font-bold text-foreground leading-tight">
            Classify Text with
            <span className="bg-gradient-primary bg-clip-text text-transparent"> AI Precision</span>
          </h1>
          
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Advanced machine learning model that instantly categorizes your text with high accuracy. 
            Perfect for sentiment analysis, content sorting, and data organization.
          </p>
        </div>
      </section>

      {/* Features */}
      <section className="container mx-auto px-4 pb-12">
        <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
          <div className="bg-card border border-border rounded-lg p-6 text-center space-y-3">
            <div className="inline-flex items-center justify-center w-12 h-12 bg-primary/10 rounded-lg">
              <Zap className="w-6 h-6 text-primary" />
            </div>
            <h3 className="font-semibold text-foreground">Lightning Fast</h3>
            <p className="text-sm text-muted-foreground">
              Get instant classification results in milliseconds
            </p>
          </div>
          
          <div className="bg-card border border-border rounded-lg p-6 text-center space-y-3">
            <div className="inline-flex items-center justify-center w-12 h-12 bg-secondary/10 rounded-lg">
              <Shield className="w-6 h-6 text-secondary" />
            </div>
            <h3 className="font-semibold text-foreground">Highly Accurate</h3>
            <p className="text-sm text-muted-foreground">
              State-of-the-art models trained on vast datasets
            </p>
          </div>
          
          <div className="bg-card border border-border rounded-lg p-6 text-center space-y-3">
            <div className="inline-flex items-center justify-center w-12 h-12 bg-accent/10 rounded-lg">
              <Sparkles className="w-6 h-6 text-accent" />
            </div>
            <h3 className="font-semibold text-foreground">Easy to Use</h3>
            <p className="text-sm text-muted-foreground">
              Simple interface, powerful results, no setup required
            </p>
          </div>
        </div>
      </section>

      {/* Demo Section */}
      <section className="container mx-auto px-4 pb-20">
        <ClassificationDemo />
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-8">
        <div className="container mx-auto px-4 text-center">
          <p className="text-sm text-muted-foreground">
            Powered by advanced AI models • Built with React & TypeScript
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
