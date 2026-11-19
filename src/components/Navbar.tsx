import { Menu, X } from 'lucide-react';
import { useState } from 'react';
import type { ActiveRoute } from '../types/nlp';

interface NavbarProps {
  activeRoute: ActiveRoute;
  onNavigate: (route: ActiveRoute) => void;
}

const NAV_ITEMS: Array<{ id: ActiveRoute; label: string }> = [
  { id: 'home', label: 'Home' },
  { id: 'dataset', label: 'Upload Dataset' },
];

export function Navbar({ activeRoute, onNavigate }: NavbarProps) {
  const [mobileOpen, setMobileOpen] = useState(false);

  const handleSelect = (route: ActiveRoute) => {
    onNavigate(route);
    setMobileOpen(false);
  };

  return (
    <header className="sticky top-0 z-50 bg-[#010A12]/85 backdrop-blur-md border-b border-[rgba(207,221,216,0.08)]">
      <div className="max-w-[1440px] mx-auto px-4 sm:px-8 h-16 flex items-center justify-between gap-8">
        {/* Zone 1: Brand Wordmark */}
        <button
          type="button"
          onClick={() => handleSelect('home')}
          className="inline-flex items-center gap-2.5 text-[#CFDDD8] hover:text-white transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          aria-label="SEMANTIQ Home"
        >
          <svg
            width="22"
            height="22"
            viewBox="0 0 24 24"
            fill="none"
            aria-hidden="true"
            className="text-[#5BA4D6] shrink-0"
          >
            <line x1="6" y1="7" x2="18" y2="6" stroke="currentColor" strokeWidth="1.4" strokeOpacity="0.65" />
            <line x1="6" y1="7" x2="12" y2="18" stroke="currentColor" strokeWidth="1.4" strokeOpacity="0.65" />
            <line x1="18" y1="6" x2="12" y2="18" stroke="currentColor" strokeWidth="1.4" strokeOpacity="0.65" />
            <circle cx="6" cy="7" r="2.2" fill="#071522" stroke="currentColor" strokeWidth="1.4" />
            <circle cx="18" cy="6" r="2.2" fill="#071522" stroke="currentColor" strokeWidth="1.4" />
            <circle cx="12" cy="18" r="2.5" fill="#143F62" stroke="#5BA4D6" strokeWidth="1.4" />
          </svg>
          <span className="font-display font-bold tracking-[0.06em] text-base">
            SEMANTIQ
          </span>
        </button>

        {/* Zone 2: Minimal Single-Line Navigation */}
        <nav aria-label="Primary Navigation" className="hidden lg:flex items-center gap-8">
          {NAV_ITEMS.map((item) => {
            const isActive = activeRoute === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => handleSelect(item.id)}
                className={`relative py-1 text-sm font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  isActive ? 'text-[#CFDDD8]' : 'text-[#91A4B0] hover:text-[#CFDDD8]'
                }`}
              >
                {item.label}
                {isActive && (
                  <span
                    aria-hidden="true"
                    className="absolute inset-x-0 -bottom-[21px] h-[2px] bg-[#5BA4D6]"
                  />
                )}
              </button>
            );
          })}
        </nav>

        {/* Zone 3: Primary CTA */}
        <div className="flex items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={() => handleSelect('dataset')}
            className="hidden sm:inline-flex items-center justify-center px-4 py-2 text-xs font-medium rounded-md bg-[#143F62] hover:bg-[#1B5278] text-[#CFDDD8] border border-[#5BA4D6]/35 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            Upload Dataset
          </button>

          <button
            type="button"
            onClick={() => setMobileOpen((o) => !o)}
            aria-label={mobileOpen ? 'Close navigation menu' : 'Open navigation menu'}
            aria-expanded={mobileOpen}
            className="lg:hidden inline-flex items-center justify-center w-9 h-9 rounded-md bg-[#071522] border border-[rgba(207,221,216,0.12)] text-[#CFDDD8] cursor-pointer"
          >
            {mobileOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {mobileOpen && (
        <div className="lg:hidden border-t border-[rgba(207,221,216,0.10)] bg-[#071522] px-4 py-4">
          <nav className="flex flex-col gap-1" aria-label="Mobile Navigation">
            {NAV_ITEMS.map((item) => {
              const isActive = activeRoute === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => handleSelect(item.id)}
                  className={`text-left px-3 py-2.5 rounded-md text-sm font-medium transition-colors cursor-pointer ${
                    isActive
                      ? 'bg-[#143F62] text-[#CFDDD8]'
                      : 'text-[#91A4B0] hover:bg-[#0B1925] hover:text-[#CFDDD8]'
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </nav>
        </div>
      )}
    </header>
  );
}
