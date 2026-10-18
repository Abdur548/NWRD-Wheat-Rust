const MetricCard = ({ title, value, isPercentage = false, note }) => {
  const number = typeof value === 'string' ? parseFloat(value) : value;
  const display = Number.isFinite(number)
    ? isPercentage
      ? `${(number * 100).toFixed(1)}%`
      : number.toFixed(3)
    : '–';

  return (
    <div className="bg-card p-6 rounded-xl shadow-sm border border-gray-100 flex flex-col justify-between">
      <h3 className="text-text-secondary font-medium mb-3">{title}</h3>
      <span className="text-4xl font-bold text-text-primary">{display}</span>
      {note && <p className="text-xs text-text-secondary mt-3">{note}</p>}
    </div>
  );
};

export default MetricCard;
