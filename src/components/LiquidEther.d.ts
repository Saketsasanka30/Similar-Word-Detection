import type { CSSProperties } from 'react';

export interface LiquidEtherProps {
  colors?: string[];
  intensity?: number;
  mouseForce?: number;
  cursorSize?: number;
  persistence?: number;
  viscosity?: number;
  resolution?: number;
  bounceEdges?: boolean;
  clickBurst?: boolean;
  autoDemo?: boolean;
  autoSpeed?: number;
  autoIntensity?: number;
  autoResumeDelay?: number;
  intro?: boolean;
  lightMode?: boolean;
  backgroundColor?: string;
  maxDevicePixelRatio?: number;
  className?: string;
  style?: CSSProperties;
}

declare const LiquidEther: (props: LiquidEtherProps) => React.JSX.Element;
export default LiquidEther;
