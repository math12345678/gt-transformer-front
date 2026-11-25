import React, { useState } from 'react';
import { AccessSection } from './components/AccessSection';
import { Footer } from './components/Footer';
import type { ScanFocusMode } from './components/GenomeScanPreview';
import { HeroSection } from './components/HeroSection';
import { MetricsStrip } from './components/MetricsStrip';
import { Navbar } from './components/Navbar';
import { PipelineSection } from './components/PipelineSection';
import { RecombinationHelix3D } from './components/RecombinationHelix3D';
import { ScanWalkthroughSection } from './components/ScanWalkthroughSection';
import { useTheme } from './hooks/useTheme';

export const App: React.FC = () => {
  const { isDark, preference, toggleTheme } = useTheme();
  const [activeScanMode, setActiveScanMode] = useState<ScanFocusMode>('flank_kl');
  const [selectedEventId, setSelectedEventId] = useState<number>(7);

  const handleJumpToHero = (eventId: number, mode: ScanFocusMode) => {
    setSelectedEventId(eventId);
    setActiveScanMode(mode);
    const overviewEl = document.getElementById('overview');
    if (overviewEl) {
      const reducedMotion =
        typeof window !== 'undefined' &&
        typeof window.matchMedia === 'function' &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      overviewEl.scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth', block: 'start' });
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#F6F6F3] dark:bg-[#121316] text-[#141416] dark:text-[#F3F4F6] transition-colors duration-200">
      <Navbar isDark={isDark} preference={preference} onToggleTheme={toggleTheme} />

      <main className="flex-1">
        <HeroSection
          activeTab={activeScanMode}
          onChangeTab={setActiveScanMode}
          selectedEventId={selectedEventId}
          onSelectEvent={setSelectedEventId}
          isDark={isDark}
        />
        <RecombinationHelix3D
          selectedEventId={selectedEventId}
          onSelectEvent={setSelectedEventId}
          isDark={isDark}
        />
        <ScanWalkthroughSection onJumpToHero={handleJumpToHero} isDark={isDark} />
        <MetricsStrip />
        <PipelineSection isDark={isDark} />
        <AccessSection />
      </main>

      <Footer />
    </div>
  );
};

export default App;
