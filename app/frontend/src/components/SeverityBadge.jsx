import { ShieldCheck, AlertTriangle, AlertCircle, XOctagon } from 'lucide-react';

// Illustrative bands for the share of the image predicted as rust (nwrd/inference.py); they are
// not a validated agronomic severity scale.
const CONFIG = {
  healthy: {
    color: 'bg-success/10 text-success border-success/20',
    icon: <ShieldCheck className="w-8 h-8" />,
    message: 'No rust detected.',
  },
  early: {
    color: 'bg-yellow-100 text-yellow-700 border-yellow-200',
    icon: <AlertTriangle className="w-8 h-8" />,
    message: 'Early: under 5% of the image is predicted as rust.',
  },
  moderate: {
    color: 'bg-accent/10 text-accent border-accent/20',
    icon: <AlertCircle className="w-8 h-8" />,
    message: 'Moderate: 5–20% of the image is predicted as rust.',
  },
  severe: {
    color: 'bg-danger/10 text-danger border-danger/20',
    icon: <XOctagon className="w-8 h-8" />,
    message: 'Severe: 20% or more of the image is predicted as rust.',
  },
};

const SeverityBadge = ({ severity }) => {
  const selected = CONFIG[severity] || CONFIG.healthy;
  return (
    <div className={`w-full p-4 rounded-xl border flex items-center gap-4 ${selected.color}`}>
      <div>{selected.icon}</div>
      <div>
        <div className="text-lg font-semibold">{selected.message}</div>
        <div className="text-xs opacity-80">
          Illustrative bands, not a validated agronomic scale; the share is of all image pixels.
        </div>
      </div>
    </div>
  );
};

export default SeverityBadge;
