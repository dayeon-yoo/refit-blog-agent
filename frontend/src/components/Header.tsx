import { useState } from 'react';

export default function Header() {
  const [logoUnavailable, setLogoUnavailable] = useState(false);
  return <header className="masthead">
    <div className="header-inner">
      <div className="brand" aria-label="RIFIT">
        {logoUnavailable
          ? <span className="brand-fallback">RIFIT</span>
          : <img src="/refit-logo.png" alt="RIFIT" className="brand-logo" onError={() => setLogoUnavailable(true)} />}
      </div>
      <span className="header-label">Blog workspace</span>
    </div>
  </header>;
}
