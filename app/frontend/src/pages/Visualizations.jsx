import { useState } from 'react';
import { imageUrl } from '../api';

const SECTIONS = [
  {
    title: 'Training curves',
    image: 'training_curves.png',
    description:
      'Validation loss and IoU per epoch for every model, mean over seeds with the range shaded. ' +
      'CANet-B4 peaks early and then overfits; the checkpoint with the best validation IoU is the ' +
      'one kept.',
  },
  {
    title: 'Confusion matrix',
    image: 'confusion_matrix.png',
    description:
      'Test pixels for the deployed checkpoint at its validation-selected threshold. TP: rust ' +
      'correctly found. FP: healthy pixels flagged as rust. FN: rust missed. TN: background ' +
      'correctly left alone, which dominates any rust dataset, so IoU is the informative summary.',
  },
  {
    title: 'Qualitative results',
    image: 'qualitative_results.png',
    description:
      "Rust-positive test patches: the image, the annotators' mask, and each model's prediction " +
      'at its own validation-selected threshold (seed 42).',
  },
];

const Figure = ({ title, image }) => {
  const [failed, setFailed] = useState(false);
  if (failed) {
    return (
      <div className="w-full h-48 flex items-center justify-center rounded-lg bg-gray-50 text-gray-400">
        {image} is not available from the backend.
      </div>
    );
  }
  return (
    <img
      src={imageUrl(image)}
      alt={title}
      className="w-full h-auto rounded-lg object-contain max-h-[600px] bg-gray-50"
      onError={() => setFailed(true)}
    />
  );
};

const Visualizations = () => (
  <div className="fade-in max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
    <h1 className="text-4xl font-extrabold mb-12 text-text-primary text-center">Figures</h1>
    <div className="space-y-16">
      {SECTIONS.map((sec) => (
        <section key={sec.image} className="flex flex-col items-center">
          <div className="w-full text-left mb-6">
            <h2 className="text-2xl font-bold text-text-primary mb-3">{sec.title}</h2>
            <p className="text-lg text-text-secondary leading-relaxed">{sec.description}</p>
          </div>
          <div className="w-full bg-card p-4 rounded-xl shadow-md border border-gray-100">
            <Figure title={sec.title} image={sec.image} />
          </div>
        </section>
      ))}
    </div>
  </div>
);

export default Visualizations;
