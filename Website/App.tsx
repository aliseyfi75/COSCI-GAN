
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import React, { useState, useEffect } from 'react';
import { SourceScene } from './components/QuantumScene';
import { ArchitectureDiagram, HeatmapComparison, AccuracyChart } from './components/Diagrams';
import { ArrowDown, Menu, X, Activity, Database, Shield, Network, GitBranch, Cpu, FileText } from 'lucide-react';

const AuthorCard = ({ name, affiliation }: { name: string, affiliation: string }) => {
  return (
    <div className="flex flex-col p-6 bg-white rounded-lg border border-slate-200 shadow-sm hover:shadow-md transition-all duration-300 w-full md:w-64">
      <div className="w-10 h-10 bg-tech-blue/10 text-tech-blue rounded-full flex items-center justify-center mb-4">
        <span className="font-mono font-bold text-lg">{name.charAt(0)}</span>
      </div>
      <h3 className="font-sans font-semibold text-lg text-slate-900 mb-1">{name}</h3>
      <p className="text-xs text-slate-500 uppercase tracking-wider font-medium">{affiliation}</p>
    </div>
  );
};

const App: React.FC = () => {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToSection = (id: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    setMenuOpen(false);
    const element = document.getElementById(id);
    if (element) {
      const headerOffset = 100;
      const elementPosition = element.getBoundingClientRect().top;
      const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

      window.scrollTo({
        top: offsetPosition,
        behavior: "smooth"
      });
    }
  };

  const paperLink = "https://proceedings.neurips.cc/paper_files/paper/2022/file/d3408794e41dd23e34634344d662f5e9-Paper-Conference.pdf";

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 selection:bg-tech-teal selection:text-white">
      
      {/* Navigation */}
      <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled ? 'bg-white/90 backdrop-blur-md shadow-sm py-3 border-b border-slate-100' : 'bg-transparent py-6'}`}>
        <div className="container mx-auto px-6 flex justify-between items-center">
          <div className="flex items-center gap-3 cursor-pointer" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
            <div className="w-8 h-8 bg-gradient-to-br from-tech-blue to-tech-teal rounded-md flex items-center justify-center text-white font-mono font-bold shadow-sm">C</div>
            <div className="flex flex-col">
              <span className="font-sans font-bold text-lg leading-none text-slate-900 tracking-tight">COSCI-GAN</span>
              <span className="text-[10px] font-mono text-slate-500 uppercase leading-none mt-1">NeurIPS 2022</span>
            </div>
          </div>
          
          <div className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-600">
            <a href="#problem" onClick={scrollToSection('problem')} className="hover:text-tech-blue transition-colors cursor-pointer">The Problem</a>
            <a href="#solution" onClick={scrollToSection('solution')} className="hover:text-tech-blue transition-colors cursor-pointer">Architecture</a>
            <a href="#results" onClick={scrollToSection('results')} className="hover:text-tech-blue transition-colors cursor-pointer">Evaluation</a>
            <a href={paperLink} target="_blank" rel="noreferrer" className="px-4 py-2 bg-slate-900 text-white rounded-md hover:bg-slate-800 transition-colors text-xs font-bold uppercase tracking-wider shadow-sm flex items-center gap-2">
              Read Paper <FileText size={14} />
            </a>
          </div>

          <button className="md:hidden text-slate-900 p-2" onClick={() => setMenuOpen(!menuOpen)}>
            {menuOpen ? <X /> : <Menu />}
          </button>
        </div>
      </nav>

      {/* Mobile Menu */}
      {menuOpen && (
        <div className="fixed inset-0 z-40 bg-white flex flex-col items-center justify-center gap-8 text-xl font-sans animate-fade-in">
            <a href="#problem" onClick={scrollToSection('problem')} className="hover:text-tech-blue transition-colors">The Problem</a>
            <a href="#solution" onClick={scrollToSection('solution')} className="hover:text-tech-blue transition-colors">Architecture</a>
            <a href="#results" onClick={scrollToSection('results')} className="hover:text-tech-blue transition-colors">Evaluation</a>
            <a href={paperLink} target="_blank" rel="noreferrer" className="text-slate-900 font-bold flex items-center gap-2">Read Paper <FileText size={16} /></a>
        </div>
      )}

      {/* Hero Section */}
      <header className="relative h-screen flex items-center overflow-hidden bg-slate-900 text-white">
        <div className="absolute inset-0 z-0">
            <SourceScene />
        </div>
        
        <div className="absolute inset-0 bg-gradient-to-r from-slate-900/90 via-slate-900/70 to-transparent pointer-events-none"></div>

        <div className="relative z-10 container mx-auto px-6 pt-20">
          <div className="max-w-3xl">
            <div className="inline-flex items-center gap-2 mb-6 px-3 py-1 rounded-full bg-tech-blue/20 border border-tech-blue/30 text-tech-blue text-xs font-mono font-bold">
                <span className="w-2 h-2 rounded-full bg-tech-blue animate-pulse"></span>
                MULTIVARIATE TIME SERIES GENERATION
            </div>
            <h1 className="font-sans font-bold text-4xl md:text-6xl lg:text-7xl leading-tight mb-6 tracking-tight">
              Synthetic Data from a <br/>
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-tech-blue to-tech-teal">Common Source</span>
            </h1>
            <p className="text-lg md:text-xl text-slate-300 font-light leading-relaxed mb-10 max-w-2xl">
              A novel framework for generating multivariate time series that originates from a single biological source. COSCI-GAN preserves complex inter-channel dynamics and correlations, outperforming state-of-the-art methods.
            </p>
            
            <div className="flex flex-wrap gap-4">
               <a href="#problem" onClick={scrollToSection('problem')} className="px-6 py-3 bg-white text-slate-900 rounded-md font-medium hover:bg-slate-100 transition-colors flex items-center gap-2">
                  Explore Framework <ArrowDown size={16} />
               </a>
               <a href="https://github.com/aliseyfi75/COSCI-GAN" target="_blank" rel="noreferrer" className="px-6 py-3 border border-slate-600 text-slate-200 rounded-md font-medium hover:bg-slate-800 transition-colors flex items-center gap-2">
                  <GitBranch size={16} /> View Code
               </a>
            </div>
          </div>
        </div>
      </header>

      <main>
        {/* The Problem Section */}
        <section id="problem" className="py-24 bg-white">
          <div className="container mx-auto px-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-16 items-center">
                <div>
                    <h2 className="text-sm font-bold text-tech-blue uppercase tracking-widest mb-2">The Motivation</h2>
                    <h3 className="text-3xl font-bold text-slate-900 mb-6">Single Source, Complex Dynamics</h3>
                    <div className="prose text-slate-600 leading-relaxed space-y-6 text-lg">
                        <p>
                            A common type of <strong>Multivariate Time Series (MTS)</strong> originates from a single source, such as the biometric measurements from a medical patient, stock prices from economic events, or seismic measurements from a single earthquake.
                        </p>
                        <p>
                           This common origin leads to specific correlation patterns and complex time dynamics across the individual time series. Capturing these patterns is crucial for machine learning models to accurately classify, predict, or perform downstream tasks.
                        </p>
                        <p>
                            However, <strong>data scarcity</strong> is a significant barrier. In biomedical and financial fields, data sharing is restricted by regulatory requirements (HIPAA, GDPR) and ethical concerns. Even anonymized data associated with a single individual can lead to privacy breaches.
                        </p>
                        <p className="border-l-4 border-tech-blue pl-4 italic text-slate-500">
                            The Limitation: Standard generation models (like typical GANs) often fail to learn the <strong>joint distribution</strong> of these coupled signals, treating channels as independent and losing the vital inter-channel dynamics.
                        </p>
                    </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                    <div className="p-6 bg-slate-50 rounded-xl border border-slate-100 hover:border-tech-blue/30 transition-colors group">
                        <Shield className="text-tech-teal mb-4 group-hover:scale-110 transition-transform" size={32} />
                        <h4 className="font-bold text-slate-900 mb-2">Privacy Constraints</h4>
                        <p className="text-sm text-slate-500">Regulatory hurdles and re-identification risks make sharing real individual data nearly impossible.</p>
                    </div>
                    <div className="p-6 bg-slate-50 rounded-xl border border-slate-100 hover:border-tech-blue/30 transition-colors group">
                        <Activity className="text-tech-blue mb-4 group-hover:scale-110 transition-transform" size={32} />
                        <h4 className="font-bold text-slate-900 mb-2">Valuable Patterns</h4>
                        <p className="text-sm text-slate-500">Complex dynamical patterns between channels contain vital information for downstream ML tasks.</p>
                    </div>
                    <div className="p-6 bg-slate-50 rounded-xl border border-slate-100 hover:border-tech-blue/30 transition-colors group">
                        <Database className="text-slate-600 mb-4 group-hover:scale-110 transition-transform" size={32} />
                        <h4 className="font-bold text-slate-900 mb-2">Data Augmentation</h4>
                        <p className="text-sm text-slate-500">Synthetic MTS can increase the contribution of underrepresented sub-populations and improve classifier performance.</p>
                    </div>
                    <div className="p-6 bg-slate-50 rounded-xl border border-slate-100 hover:border-tech-blue/30 transition-colors group">
                        <Network className="text-tech-purple mb-4 group-hover:scale-110 transition-transform" size={32} />
                        <h4 className="font-bold text-slate-900 mb-2">Joint Distribution</h4>
                        <p className="text-sm text-slate-500">Standard GANs struggle to learn the hard inter-channel/feature correlations inherent in single-source data.</p>
                    </div>
                </div>
            </div>
          </div>
        </section>

        {/* The Solution: COSCI-GAN Architecture */}
        <section id="solution" className="py-24 bg-slate-900 text-white overflow-hidden relative">
            {/* Background Pattern */}
            <div className="absolute inset-0 opacity-5 pointer-events-none">
                 <div className="absolute inset-0" style={{backgroundImage: 'radial-gradient(#4b5563 1px, transparent 1px)', backgroundSize: '30px 30px'}}></div>
            </div>

            <div className="container mx-auto px-6 relative z-10">
                <div className="text-center max-w-3xl mx-auto mb-16">
                    <h2 className="text-sm font-bold text-tech-teal uppercase tracking-widest mb-2">The Framework</h2>
                    <h3 className="text-3xl md:text-4xl font-bold mb-6">Common Source Coordinated GAN</h3>
                    <p className="text-slate-400 text-lg">
                        Our method separates the generation task: Channel GANs focus on the marginal distribution of each sensor, while a Central Discriminator ensures the conditional distributions (correlations) are realistic.
                    </p>
                </div>

                <ArchitectureDiagram />
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-16 text-center">
                     <div>
                        <div className="inline-block p-3 rounded-full bg-slate-800 text-tech-blue mb-4 border border-slate-700">
                            <Cpu size={24} />
                        </div>
                        <h4 className="font-bold text-xl mb-2">Channel GANs</h4>
                        <p className="text-slate-400 text-sm leading-relaxed">
                            Dedicated Generator-Discriminator pairs ($G_i, D_i$) for each channel. They learn the <strong>marginal distribution</strong> of individual time series, ensuring high signal quality for each channel.
                        </p>
                     </div>
                     <div>
                        <div className="inline-block p-3 rounded-full bg-slate-800 text-tech-teal mb-4 border border-slate-700">
                            <GitBranch size={24} />
                        </div>
                        <h4 className="font-bold text-xl mb-2">Common Noise Source</h4>
                        <p className="text-slate-400 text-sm leading-relaxed">
                            All generators share the same initial noise vector $z$. This latent space represents the patient's "whole biological environment," acting as the common source of variation.
                        </p>
                     </div>
                     <div>
                        <div className="inline-block p-3 rounded-full bg-slate-800 text-tech-purple mb-4 border border-slate-700">
                            <Network size={24} />
                        </div>
                        <h4 className="font-bold text-xl mb-2">Central Discriminator</h4>
                        <p className="text-slate-400 text-sm leading-relaxed">
                            Receives the concatenated multivariate time series. It focuses on the <strong>conditional distributions</strong> and penalizes the model if inter-channel dynamics and dependencies are lost.
                        </p>
                     </div>
                </div>
            </div>
        </section>

        {/* Evaluation Results */}
        <section id="results" className="py-24 bg-slate-50">
            <div className="container mx-auto px-6">
                <div className="mb-16">
                    <h2 className="text-sm font-bold text-tech-blue uppercase tracking-widest mb-2">Evaluation</h2>
                    <h3 className="text-3xl font-bold text-slate-900 mb-6">Outperforming State-of-the-Art</h3>
                    <p className="max-w-2xl text-slate-600 mb-8">
                        We evaluated COSCI-GAN on the <strong>EEG Eye State Dataset</strong> (14 channels). We compared performance against **TimeGAN** (NeurIPS 2019) and **Fourier Flows** (ICLR 2021) in both correlation preservation and downstream utility.
                    </p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 mb-16">
                    <div>
                        <h4 className="font-bold text-xl text-slate-800 mb-4 flex items-center gap-2">
                            <span className="w-2 h-6 bg-tech-blue rounded-sm"></span>
                            Correlation Analysis (Catch22)
                        </h4>
                        <p className="text-slate-600 mb-6 text-sm">
                            Using <strong>Catch22</strong> canonical time-series features, we analyzed pairwise correlations. The heatmaps below demonstrate that without the Central Discriminator (CD), the complex "block-like" dependency patterns of real EEG data are destroyed. COSCI-GAN preserves these statistical properties faithfully.
                        </p>
                        <HeatmapComparison />
                    </div>
                    <div>
                         <h4 className="font-bold text-xl text-slate-800 mb-4 flex items-center gap-2">
                            <span className="w-2 h-6 bg-tech-teal rounded-sm"></span>
                            Eye Blink Detection (Robustness)
                        </h4>
                        <p className="text-slate-600 mb-6 text-sm">
                            We trained an LSTM classifier to detect eye blinks. In <strong>Data Augmentation</strong> experiments (adding synthetic samples to the real dataset), COSCI-GAN yielded significantly higher accuracy and robustness compared to baselines, especially in low-data regimes.
                        </p>
                        <AccuracyChart />
                    </div>
                </div>
            </div>
        </section>
        
        {/* Paper & Authors */}
        <section id="paper" className="py-24 bg-white border-t border-slate-200">
            <div className="container mx-auto px-6">
                 <h2 className="text-3xl font-bold text-slate-900 mb-12 text-center">The Research Team</h2>
                 
                 <div className="flex flex-wrap justify-center gap-8 mb-16">
                    <AuthorCard name="Ali Seyfi" affiliation="Dept. of Computer Science, UBC" />
                    <AuthorCard name="Jean-Francois Rajotte" affiliation="Data Science Institute, UBC" />
                    <AuthorCard name="Raymond T. Ng" affiliation="Dept. of Computer Science, UBC" />
                 </div>

                 <div className="bg-slate-900 text-slate-300 rounded-2xl p-8 md:p-12 flex flex-col md:flex-row items-center justify-between gap-8">
                    <div>
                        <h3 className="text-2xl font-bold text-white mb-2">NeurIPS 2022</h3>
                        <p className="opacity-80">Proceedings of the 36th Conference on Neural Information Processing Systems</p>
                    </div>
                    <div className="flex gap-4">
                        <a href={paperLink} target="_blank" rel="noreferrer" className="px-8 py-3 bg-white text-slate-900 font-bold rounded-lg hover:bg-slate-100 transition-colors shadow-lg">
                            View Proceedings
                        </a>
                        <a href="https://github.com/aliseyfi75/COSCI-GAN" target="_blank" rel="noreferrer" className="px-8 py-3 border border-slate-600 text-white font-bold rounded-lg hover:bg-slate-800 transition-colors">
                            GitHub Code
                        </a>
                    </div>
                 </div>
            </div>
        </section>

      </main>

      <footer className="bg-slate-950 text-slate-500 py-12 border-t border-slate-900">
        <div className="container mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-6">
            <div className="text-center md:text-left">
                <div className="text-white font-sans font-bold text-xl mb-1">COSCI-GAN</div>
                <p className="text-xs">Generating multivariate time series with Common Source Coordinated GAN</p>
            </div>
            <div className="text-xs">
                &copy; 2022 Seyfi et al. University of British Columbia.
            </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
