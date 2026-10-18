import { useState, useEffect } from 'react';
import { api } from '../api';
import MetricCard from '../components/MetricCard';

const Results = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .get('/api/results')
      .then((res) => setData(res.data))
      .catch((error) => console.error('Failed to fetch results:', error))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-[50vh]">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }
  if (!data) {
    return (
      <div className="text-center text-danger mt-20">
        Failed to load results. Ensure the backend is running.
      </div>
    );
  }

  const { final_metrics: fm = {}, best_threshold, best_epoch, seed, history = {} } = data;
  const epochs = (history.train_loss || []).map((trainLoss, i) => ({
    epoch: i + 1,
    trainLoss,
    valLoss: history.val_loss?.[i],
    valIou: history.val_iou?.[i],
    valDice: history.val_dice?.[i],
  }));

  return (
    <div className="fade-in max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <h1 className="text-4xl font-extrabold mb-2 text-text-primary">Results</h1>
      <p className="text-text-secondary mb-8 max-w-4xl">
        Test split (10 held-out photographs) for the deployed checkpoint: CANet-B4 seed {seed},
        epoch {best_epoch}, the seed with the best validation IoU. Metrics are pooled over every
        test pixel. The mean over three seeds, and the comparison with the baselines, are in{' '}
        <code className="text-sm">docs/results.md</code>.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-4">
        <MetricCard title="IoU" value={fm['IoU']} />
        <MetricCard title="Dice (F1)" value={fm['F1 / Dice']} />
        <MetricCard title="Precision" value={fm['Precision']} />
        <MetricCard title="Recall" value={fm['Recall']} />
        <MetricCard title="Specificity" value={fm['Specificity']} isPercentage />
        <MetricCard title="Accuracy" value={fm['Accuracy']} isPercentage />
      </div>
      <p className="text-xs text-text-secondary mb-12">
        Background pixels dominate the test split, so accuracy and specificity are high for any
        model; IoU and Dice are the informative measures.
      </p>

      <div className="bg-card p-6 rounded-xl shadow-sm border border-gray-100 mb-12 flex items-center justify-between gap-6">
        <div>
          <h3 className="text-lg font-bold text-text-primary">Binarisation threshold</h3>
          <p className="text-text-secondary mt-1">
            Chosen to maximise IoU on the validation split, then applied unchanged to the test
            split.
          </p>
        </div>
        <div className="bg-primary/10 text-primary text-3xl font-extrabold px-6 py-3 rounded-lg border border-primary/20">
          {best_threshold}
        </div>
      </div>

      <h2 className="text-2xl font-bold mb-2 text-text-primary">Training history</h2>
      <p className="text-text-secondary mb-4">
        Validation IoU and Dice per epoch at threshold 0.5, pooled over all pixels.
      </p>
      <div className="bg-card rounded-xl shadow-sm border border-gray-200 overflow-hidden mb-12">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-200">
              {['Epoch', 'Train loss', 'Val loss', 'Val IoU', 'Val Dice'].map((h) => (
                <th key={h} className="p-4 font-semibold text-text-secondary">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {epochs.map((row) => (
              <tr
                key={row.epoch}
                className={row.epoch === best_epoch ? 'bg-primary/10 font-semibold' : 'even:bg-gray-50/50'}
              >
                <td className="p-4 text-text-primary">{row.epoch}</td>
                <td className="p-4 text-text-secondary">{row.trainLoss?.toFixed(4)}</td>
                <td className="p-4 text-text-secondary">{row.valLoss?.toFixed(4)}</td>
                <td className="p-4 text-text-secondary">{row.valIou?.toFixed(4)}</td>
                <td className="p-4 text-text-secondary">{row.valDice?.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Results;
