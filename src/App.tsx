import { useState } from 'react';
import { DatasetAnalyzer } from './components/DatasetAnalyzer';
import { HeroSection } from './components/HeroSection';
import LiquidEther from './components/LiquidEther';
import { Navbar } from './components/Navbar';
import type { ActiveRoute } from './types/nlp';

export default function App() {
  const [activeRoute, setActiveRoute] = useState<ActiveRoute>('home');

  const navigateTo = (route: ActiveRoute) => {
    setActiveRoute(route);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const scrollToWorkspace = () => {
    if (activeRoute !== 'home') {
      setActiveRoute('home');
    }
    window.setTimeout(() => {
      document.getElementById('workspace')?.scrollIntoView({ behavior: 'smooth' });
    }, 40);
  };

  return (
    <div className="relative min-h-screen flex flex-col bg-[#010A12] text-[#CFDDD8] overflow-x-hidden">
      {/* React Bits LiquidEther Whole-App Background */}
      <div
        aria-hidden="true"
        className="fixed inset-0 z-0 pointer-events-none"
      >
        <LiquidEther
          colors={['#5227FF', '#FF9FFC', '#B497CF']}
          mouseForce={20}
          cursorSize={100}
          persistence={0.75}
          autoDemo={true}
          autoSpeed={0.5}
          autoIntensity={2.2}
          backgroundColor="#010A12"
        />
        {/* Subtle readability scrim preserving deep-navy contrast */}
        <div className="absolute inset-0 bg-[#010A12]/55" />
      </div>

      {/* Foreground Application Layer */}
      <div className="relative z-10 flex-1 flex flex-col">
        <Navbar activeRoute={activeRoute} onNavigate={navigateTo} />

        <main className="flex-1">
          {activeRoute === 'home' && (
            <>
              <HeroSection onUploadDataset={scrollToWorkspace} />

              {/* Upload Dataset Workspace */}
              <section id="workspace" className="max-w-[1200px] mx-auto px-4 sm:px-8 py-14 sm:py-20">
                <div className="mb-8">
                  <h2 className="font-display font-bold text-2xl sm:text-3xl tracking-tight text-[#CFDDD8]">
                    Upload Dataset
                  </h2>
                  <p className="text-sm text-[#91A4B0] mt-1">
                    Upload a text dataset or use the training corpus to predict similar words using cosine similarity.
                  </p>
                </div>

                <DatasetAnalyzer embedded />
              </section>
            </>
          )}

          {activeRoute === 'dataset' && <DatasetAnalyzer />}
        </main>

        <footer className="border-t border-[rgba(207,221,216,0.08)] bg-[#010A12]/70 backdrop-blur-sm">
          <div className="max-w-[1440px] mx-auto px-4 sm:px-8 py-8 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div className="font-display font-bold tracking-[0.06em] text-sm text-[#CFDDD8]">
              SEMANTIQ
            </div>

            <nav
              aria-label="Footer Navigation"
              className="flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-[#91A4B0]"
            >
              <button type="button" onClick={() => navigateTo('home')} className="hover:text-[#CFDDD8] cursor-pointer">
                Home
              </button>
              <button type="button" onClick={() => navigateTo('dataset')} className="hover:text-[#CFDDD8] cursor-pointer">
                Upload Dataset
              </button>
            </nav>
          </div>
        </footer>
      </div>
    </div>
  );
}
