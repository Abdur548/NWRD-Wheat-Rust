import { Link } from 'react-router-dom';
import { Database, Layers, Cpu, Target, ArrowRight } from 'lucide-react';

const CARDS = [
  { icon: Database, title: 'Dataset', value: 'NWRD (NUST Wheat Rust Disease)' },
  { icon: Layers, title: 'Architecture', value: 'CANet' },
  { icon: Cpu, title: 'Encoder', value: 'EfficientNet-B4' },
  { icon: Target, title: 'Task', value: 'Binary segmentation' },
];

const Landing = () => {
  return (
    <div className="fade-in pb-12">
      <section className="w-full bg-gradient-to-br from-primary to-primary-light text-white py-24 px-4 text-center">
        <h1 className="text-5xl md:text-6xl font-extrabold mb-6 tracking-tight">
          Wheat Rust Segmentation
        </h1>
        <p className="text-xl md:text-2xl font-medium opacity-90 max-w-3xl mx-auto">
          Pixel-level stripe-rust segmentation of field photographs with CANet-B4
        </p>
        <div className="mt-10 flex justify-center gap-4">
          <Link
            to="/predict"
            className="bg-white text-primary px-8 py-3 rounded-full font-bold text-lg hover:bg-gray-100 transition shadow-lg flex items-center gap-2"
          >
            Try live prediction <ArrowRight className="w-5 h-5" />
          </Link>
          <Link
            to="/results"
            className="bg-transparent border-2 border-white text-white px-8 py-3 rounded-full font-bold text-lg hover:bg-white/10 transition shadow-lg"
          >
            View results
          </Link>
        </div>
      </section>

      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 -mt-10">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {CARDS.map(({ icon: Icon, title, value }) => (
            <div
              key={title}
              className="bg-card p-6 rounded-xl shadow-lg border border-gray-100 flex items-center gap-4"
            >
              <div className="bg-primary/10 p-3 rounded-full">
                <Icon className="w-8 h-8 text-primary" />
              </div>
              <div>
                <h3 className="text-text-secondary text-sm font-medium">{title}</h3>
                <p className="text-text-primary text-lg font-bold">{value}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 mt-16 text-text-secondary leading-relaxed">
        <h2 className="text-2xl font-bold text-text-primary mb-3">How it compares</h2>
        <p>
          CANet-B4 was compared with the NWRD paper&apos;s UNet and with DeepLabV3+ R50 under one
          protocol: the same data split by photograph, training schedule and threshold rule, three
          seeds each. On 512px patches its accuracy is comparable to both baselines, and at the
          paper&apos;s 128px scale it is less accurate than UNet. Its advantage is efficiency: about
          a tenth of UNet&apos;s computation. The full comparison is in{' '}
          <code className="text-sm">docs/results.md</code>.
        </p>
      </section>

      <footer className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-16 text-center text-text-secondary border-t border-gray-200 pt-8">
        <p className="font-medium mb-2">
          Project team: Syed Muhammad Abdur Rahman (467471), Abdul Hadi Sheikh (454448), Abdullah
          Salim Nizami (457223)
        </p>
        <p>Supervisor: Dr. Imran Malik</p>
        <p className="mt-4 text-sm opacity-75">NUST SEECS — April 2026</p>
      </footer>
    </div>
  );
};

export default Landing;
