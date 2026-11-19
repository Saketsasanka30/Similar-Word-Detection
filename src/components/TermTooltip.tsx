import { HelpCircle } from 'lucide-react';
import { useState } from 'react';
import { NLP_GLOSSARY } from '../utils/tooltips';

interface TermTooltipProps {
  termKey: keyof typeof NLP_GLOSSARY;
  children?: React.ReactNode;
}

export function TermTooltip({ termKey, children }: TermTooltipProps) {
  const [open, setOpen] = useState(false);
  const entry = NLP_GLOSSARY[termKey];
  if (!entry) return <>{children ?? termKey}</>;

  return (
    <span
      className="relative inline-flex items-center gap-1 group"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <span className="border-b border-dotted border-[#5BA4D6]/50 text-[#CFDDD8]">
        {children ?? entry.term}
      </span>
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        aria-label={`Explain ${entry.term}`}
        className="inline-flex items-center justify-center text-[#91A4B0] hover:text-[#5BA4D6] transition-colors focus-visible:outline-2"
      >
        <HelpCircle className="w-3.5 h-3.5" />
      </button>

      {open && (
        <span
          role="tooltip"
          className="absolute left-0 bottom-full mb-2 z-40 w-64 sm:w-72 p-3 rounded-md bg-[#071522] border border-[rgba(207,221,216,0.18)] text-xs leading-relaxed text-[#91A4B0] shadow-xl pointer-events-none"
        >
          <span className="block font-display font-semibold text-[#CFDDD8] mb-1">
            {entry.term}
          </span>
          {entry.short}
        </span>
      )}
    </span>
  );
}
