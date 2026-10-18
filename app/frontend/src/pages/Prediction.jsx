import { useState, useEffect } from 'react';
import { UploadCloud, Image as ImageIcon, Download, Loader2 } from 'lucide-react';
import { api } from '../api';
import SeverityBadge from '../components/SeverityBadge';

// The threshold grid searched on validation (nwrd/config.py).
const THRESHOLD_MIN = 0.05;
const THRESHOLD_MAX = 0.95;

const errorMessage = (err) => {
  const detail = err.response?.data?.detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg).join('; ');
  return detail || 'An error occurred during prediction.';
};

const Prediction = () => {
  const [threshold, setThreshold] = useState(0.5);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Start the slider at the threshold selected on the validation split.
  useEffect(() => {
    api
      .get('/api/results')
      .then((res) => {
        const t = Number(res.data?.best_threshold);
        if (t >= THRESHOLD_MIN && t <= THRESHOLD_MAX) setThreshold(t);
      })
      .catch(() => {});
  }, []);

  // Release the previous preview's memory when it is replaced or the page closes.
  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl]);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handlePredict = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('threshold', threshold);
    try {
      const res = await api.post('/api/predict', formData);
      setResult(res.data);
    } catch (err) {
      console.error(err);
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadMask = () => {
    if (!result?.pred_mask_b64) return;
    const link = document.createElement('a');
    link.href = result.pred_mask_b64;
    link.download = `mask_${selectedFile.name.replace(/\.[^.]+$/, '')}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="fade-in max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col md:flex-row gap-8">
      {/* Sidebar */}
      <div className="w-full md:w-80 flex-shrink-0">
        <div className="bg-card p-6 rounded-xl shadow-md border border-gray-100 sticky top-24">
          <h2 className="text-xl font-bold mb-6 text-text-primary">Settings</h2>

          <div className="mb-8">
            <label htmlFor="threshold" className="block text-sm font-medium text-text-secondary mb-2">
              Detection threshold
            </label>
            <input
              id="threshold"
              type="range"
              min={THRESHOLD_MIN}
              max={THRESHOLD_MAX}
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full accent-primary"
            />
            <div className="mt-2 text-center text-3xl font-extrabold text-primary">
              {threshold.toFixed(2)}
            </div>
            <p className="text-xs text-text-secondary text-center mt-2">
              Lower finds more rust; higher is more conservative. It starts at the threshold chosen
              on validation.
            </p>
          </div>

          <div className="mb-6">
            <span className="block text-sm font-medium text-text-secondary mb-2">Upload image</span>
            <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 transition">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <UploadCloud className="w-8 h-8 text-gray-400 mb-2" />
                <p className="text-sm text-gray-500">
                  <span className="font-semibold">Click to upload</span>
                </p>
                <p className="text-xs text-gray-400 mt-1">JPG, PNG or TIFF photograph</p>
              </div>
              <input
                type="file"
                className="hidden"
                accept=".jpg,.jpeg,.png,.tif,.tiff"
                onChange={handleFileChange}
              />
            </label>
          </div>

          <button
            onClick={handlePredict}
            disabled={!selectedFile || loading}
            className={`w-full py-3 rounded-lg font-bold text-white shadow-md transition flex justify-center items-center gap-2 ${
              !selectedFile || loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-primary hover:bg-primary-light'
            }`}
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" /> Analysing image...
              </>
            ) : (
              'Run inference'
            )}
          </button>
          <p className="text-xs text-text-secondary mt-3">
            The photograph is predicted at native resolution in 512px patches, as the test set was
            scored: about 1.5 s per patch on a CPU.
          </p>
        </div>
      </div>

      {/* Main panel */}
      <div className="flex-grow">
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">
            {error}
          </div>
        )}

        {!selectedFile && !result && (
          <div className="h-[60vh] flex flex-col items-center justify-center border-2 border-dashed border-gray-200 rounded-2xl bg-gray-50">
            <ImageIcon className="w-24 h-24 text-gray-300 mb-4" />
            <h3 className="text-xl font-bold text-gray-400">Upload an image to begin</h3>
          </div>
        )}

        {selectedFile && !result && !loading && (
          <div className="h-[60vh] flex flex-col items-center justify-center border-2 border-dashed border-gray-200 rounded-2xl bg-gray-50">
            <img src={previewUrl} alt="Preview" className="max-h-64 object-contain rounded-lg shadow-sm mb-6" />
            <h3 className="text-xl font-bold text-gray-600">Ready for inference</h3>
            <p className="text-gray-400">Click Run inference</p>
          </div>
        )}

        {result && (
          <div className="fade-in space-y-6">
            <SeverityBadge severity={result.severity} />

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-card p-4 rounded-xl border shadow-sm text-center">
                <p className="text-text-secondary text-sm font-medium">Rust pixels</p>
                <p className="text-2xl font-bold text-danger">{result.rust_pixels.toLocaleString()}</p>
              </div>
              <div className="bg-card p-4 rounded-xl border shadow-sm text-center">
                <p className="text-text-secondary text-sm font-medium">Image pixels</p>
                <p className="text-2xl font-bold text-text-primary">{result.total_pixels.toLocaleString()}</p>
              </div>
              <div className="bg-card p-4 rounded-xl border shadow-sm text-center">
                <p className="text-text-secondary text-sm font-medium">Predicted rust</p>
                <p className="text-2xl font-bold text-primary">{result.coverage_pct}%</p>
              </div>
            </div>
            <p className="text-sm text-text-secondary">
              {result.width}×{result.height} px, {result.patches} patches of 512 px, predicted in{' '}
              {result.seconds} s at threshold {result.threshold}.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {[
                { title: 'Original', src: result.original_b64 },
                { title: 'Probability map', src: result.prob_map_b64 },
                { title: 'Binary mask', src: result.pred_mask_b64 },
                { title: 'Overlay', src: result.overlay_b64 },
              ].map((img) => (
                <div key={img.title} className="bg-card p-4 rounded-xl border shadow-sm overflow-hidden">
                  <h4 className="font-semibold text-text-primary mb-3 text-center">{img.title}</h4>
                  <img src={img.src} alt={img.title} className="w-full h-auto rounded-lg" />
                </div>
              ))}
            </div>

            <div className="flex justify-center mt-8">
              <button
                onClick={handleDownloadMask}
                className="bg-primary text-white px-6 py-3 rounded-lg font-bold shadow-md hover:bg-primary-light transition flex items-center gap-2"
              >
                <Download className="w-5 h-5" /> Download binary mask (full resolution)
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Prediction;
