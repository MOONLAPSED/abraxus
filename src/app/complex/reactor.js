import React, { useState, useEffect } from 'react';

// Enum for quantum state
const QuantumState = {
  SUPERPOSITION: 'SUPERPOSITION',
  ENTANGLED: 'ENTANGLED',
  COLLAPSED: 'COLLAPSED'
};

// Enum for morphology
const Morphology = {
  STATIC: 0,
  DYNAMIC: 1
};

// ByteWord class implementation
class ByteWord {
  constructor(raw) {
    if (raw < 0 || raw > 255) {
      throw new Error("ByteWord must be an 8-bit integer (0-255)");
    }
    this.raw = raw;
    this.value = raw & 0xFF;
    this.state_data = (raw >> 4) & 0x0F;    // T: High nibble (4 bits)
    this.morphism = (raw >> 1) & 0x07;      // V: Middle 3 bits
    this.floor_morphic = raw & 0x01;        // C: Least significant bit
    this._refcount = 1;
    this._state = QuantumState.SUPERPOSITION;
  }

  get _pointable() {
    return this.floor_morphic === Morphology.DYNAMIC;
  }

  toString() {
    return `${this.state_data.toString(2).padStart(4, '0')}|${this.morphism.toString(2).padStart(3, '0')}${this.floor_morphic}`;
  }

  toHex() {
    return `0x${this.value.toString(16).padStart(2, '0')}`;
  }

  static xnor(a, b, width = 4) {
    return ~(a ^ b) & ((1 << width) - 1);
  }

  static abelianTransform(t, v, c) {
    if (c === 1) {
      return ByteWord.xnor(t, v);  // Apply XNOR transformation
    }
    return t;  // Identity morphism when c = 0
  }

  static fromString(binStr) {
    // Parse a string like "0010|1001"
    binStr = binStr.replace(/[^01]/g, '');
    if (binStr.length !== 8) {
      throw new Error("Binary string must be 8 bits");
    }
    return new ByteWord(parseInt(binStr, 2));
  }

  // Deep clone a ByteWord
  clone() {
    const clone = new ByteWord(this.value);
    clone._refcount = this._refcount;
    clone._state = this._state;
    return clone;
  }

  // Transform based on morphism selector
  transform(targetWord) {
    // Clone the target to avoid unintended side effects
    const target = targetWord.clone();
    
    switch(this.morphism) {
      case 0: // Identity transform
        return target;
      case 1: // Copy transform
        return new ByteWord(this.value);
      case 2: // Increment transform
        return new ByteWord((target.value + 1) & 0xFF);
      case 3: // XNOR transform
        const newT = ByteWord.abelianTransform(
          target.state_data, 
          this.state_data, 
          this.floor_morphic
        );
        const newV = target.morphism;
        const newC = target.floor_morphic;
        return new ByteWord((newT << 4) | (newV << 1) | newC);
      case 4: // Toggle control bit
        return new ByteWord(target.value ^ 0x01);
      case 5: // Swap nibbles
        const high = target.state_data;
        const low = (target.morphism << 1) | target.floor_morphic;
        return new ByteWord((low << 4) | high);
      case 6: // Bitwise NOT
        return new ByteWord(~target.value & 0xFF);
      case 7: // Random transform
        return new ByteWord(Math.floor(Math.random() * 256));
      default:
        return target;
    }
  }
}

// Self-replicating pattern simulation
class QuineSystem {
  constructor(initialByteWords) {
    // Make deep copies of the initial ByteWords to avoid reference issues
    this.byteWords = initialByteWords.map(bw => bw instanceof ByteWord ? bw.clone() : new ByteWord(bw));
    this.replicationHistory = [this.byteWords.map(bw => bw.value)];
    this.currentStep = 0;
    this.transformationLog = [];
  }

  step() {
    if (this.byteWords.length === 0) return [];
    
    // Create a fresh log for this step
    const stepLog = [];
    
    // Create a copy of the current state
    const newByteWords = this.byteWords.map(bw => bw.clone());
    
    // Process each ByteWord based on its morphism
    for (let i = 0; i < this.byteWords.length; i++) {
      const current = this.byteWords[i];
      const targetIndex = (i + 1) % this.byteWords.length; // Point to next ByteWord
      const target = this.byteWords[targetIndex];
      
      // Log the current action
      const action = {
        sourceIndex: i,
        targetIndex,
        sourceBefore: current.value,
        targetBefore: target.value,
        morphism: current.morphism,
        operation: getMorphismDescription(current.morphism)
      };
      
      // Apply transformation based on the current ByteWord's morphism
      if (current.morphism === 1) { // Copy transform
        // Add a copy of the current ByteWord to the end (NOT the target)
        const newByteWord = new ByteWord(current.value);
        newByteWords.push(newByteWord);
        action.result = 'copy';
        action.newValue = newByteWord.value;
      } else {
        // Apply other transformations on the target
        const transformedByteWord = current.transform(target);
        newByteWords[targetIndex] = transformedByteWord;
        action.result = 'transform';
        action.targetAfter = transformedByteWord.value;
      }
      
      stepLog.push(action);
    }
    
    // Update system state
    this.byteWords = newByteWords;
    this.replicationHistory.push(this.byteWords.map(bw => bw.value));
    this.transformationLog.push(stepLog);
    this.currentStep++;
    
    return this.byteWords;
  }

  reset(initialByteWords) {
    // Make deep copies of the initial ByteWords
    this.byteWords = initialByteWords.map(bw => bw instanceof ByteWord ? bw.clone() : new ByteWord(bw));
    this.replicationHistory = [this.byteWords.map(bw => bw.value)];
    this.transformationLog = [];
    this.currentStep = 0;
  }
  
  // Get detailed logs of transformations
  getTransformationLogs() {
    return this.transformationLog;
  }
  
  // Analyze growth patterns
  analyzeGrowth() {
    const growthRates = [];
    for (let i = 1; i < this.replicationHistory.length; i++) {
      const previousCount = this.replicationHistory[i-1].length;
      const currentCount = this.replicationHistory[i].length;
      const growthRate = currentCount - previousCount;
      growthRates.push(growthRate);
    }
    return growthRates;
  }
}

// Transform description based on morphism value
const getMorphismDescription = (morphism) => {
  switch(morphism) {
    case 0: return "Identity (No Change)";
    case 1: return "Copy Transform";
    case 2: return "Increment";
    case 3: return "XNOR Transform";
    case 4: return "Toggle Control";
    case 5: return "Swap Nibbles";
    case 6: return "Bitwise NOT";
    case 7: return "Random Transform";
    default: return "Unknown";
  }
};

const ByteWordDisplay = ({ byteWord, index, highlightColor = null, showDetails = false }) => {
  const style = {
    border: '2px solid #0066cc',
    borderRadius: '5px',
    padding: '8px',
    margin: '5px',
    minWidth: '120px',
    backgroundColor: highlightColor || '#fff',
    position: 'relative'
  };

  return (
    <div style={style}>
      <div style={{ position: 'absolute', top: '-20px', fontSize: '12px' }}>
        ByteWord {String.fromCharCode(65 + index)} ({byteWord.value})
      </div>
      <div style={{ fontFamily: 'Courier New', fontSize: '16px', textAlign: 'center' }}>
        {byteWord.toString()}
      </div>
      <div style={{ fontSize: '10px', marginTop: '5px', textAlign: 'center' }}>
        T:{byteWord.state_data} V:{byteWord.morphism} C:{byteWord.floor_morphic}
      </div>
      <div style={{ fontSize: '12px', marginTop: '5px', textAlign: 'center' }}>
        {getMorphismDescription(byteWord.morphism)}
      </div>
      {showDetails && (
        <div style={{ fontSize: '10px', marginTop: '5px', textAlign: 'center', color: '#555' }}>
          Hex: {byteWord.toHex()}, Pointable: {byteWord._pointable ? 'Yes' : 'No'}
        </div>
      )}
    </div>
  );
};

const ByteWordEditor = ({ byteWord, onChange }) => {
  const [stateData, setStateData] = useState(byteWord.state_data);
  const [morphism, setMorphism] = useState(byteWord.morphism);
  const [floorMorphic, setFloorMorphic] = useState(byteWord.floor_morphic);

  const handleStateDataChange = (e) => {
    const value = parseInt(e.target.value, 10);
    if (isNaN(value) || value < 0 || value > 15) return;
    setStateData(value);
    updateByteWord(value, morphism, floorMorphic);
  };

  const handleMorphismChange = (e) => {
    const value = parseInt(e.target.value, 10);
    if (isNaN(value) || value < 0 || value > 7) return;
    setMorphism(value);
    updateByteWord(stateData, value, floorMorphic);
  };

  const handleFloorMorphicChange = (e) => {
    const value = parseInt(e.target.value, 10);
    if (isNaN(value) || (value !== 0 && value !== 1)) return;
    setFloorMorphic(value);
    updateByteWord(stateData, morphism, value);
  };
  
  const handleBinaryInput = (e) => {
    try {
      const binStr = e.target.value.replace(/[^01|]/g, '');
      if (binStr.length === 8 || binStr.length === 9) { // Allow for the separator
        const byteWord = ByteWord.fromString(binStr);
        setStateData(byteWord.state_data);
        setMorphism(byteWord.morphism);
        setFloorMorphic(byteWord.floor_morphic);
        onChange(byteWord);
      }
    } catch (error) {
      console.error("Invalid binary input", error);
    }
  };

  const updateByteWord = (t, v, c) => {
    const newValue = (t << 4) | (v << 1) | c;
    onChange(new ByteWord(newValue));
  };

  return (
    <div className="flex flex-col p-3 border border-gray-300 rounded">
      <div className="mb-2">
        <label className="block text-sm font-medium">Binary Representation:</label>
        <input
          type="text"
          placeholder="TTTT|VVVC (e.g. 0010|1001)"
          value={`${stateData.toString(2).padStart(4, '0')}|${morphism.toString(2).padStart(3, '0')}${floorMorphic}`}
          onChange={handleBinaryInput}
          className="w-full p-1 border border-gray-300 rounded font-mono"
        />
      </div>
      <div className="mb-2">
        <label className="block text-sm font-medium">State Data (T - 4 bits):</label>
        <input
          type="number"
          min="0"
          max="15"
          value={stateData}
          onChange={handleStateDataChange}
          className="w-full p-1 border border-gray-300 rounded"
        />
      </div>
      <div className="mb-2">
        <label className="block text-sm font-medium">Morphism (V - 3 bits):</label>
        <select
          value={morphism}
          onChange={handleMorphismChange}
          className="w-full p-1 border border-gray-300 rounded"
        >
          <option value="0">0 - Identity</option>
          <option value="1">1 - Copy</option>
          <option value="2">2 - Increment</option>
          <option value="3">3 - XNOR</option>
          <option value="4">4 - Toggle Control</option>
          <option value="5">5 - Swap Nibbles</option>
          <option value="6">6 - Bitwise NOT</option>
          <option value="7">7 - Random</option>
        </select>
      </div>
      <div className="mb-2">
        <label className="block text-sm font-medium">Control Bit (C - 1 bit):</label>
        <select
          value={floorMorphic}
          onChange={handleFloorMorphicChange}
          className="w-full p-1 border border-gray-300 rounded"
        >
          <option value="0">0 - Static</option>
          <option value="1">1 - Dynamic</option>
        </select>
      </div>
    </div>
  );
};

// Add a component to visualize transformations
const TransformationVisualizer = ({ quineSystem }) => {
  const logs = quineSystem.getTransformationLogs();
  
  if (!logs || logs.length === 0) {
    return <div className="text-center text-gray-500">No transformation logs available yet.</div>;
  }
  
  return (
    <div className="mt-4">
      <h3 className="text-lg font-semibold mb-2">Transformation Logs</h3>
      <div className="overflow-x-auto">
        {logs.map((stepLog, stepIndex) => (
          <div key={stepIndex} className="mb-4 border-b pb-2">
            <h4 className="font-medium">Step {stepIndex + 1}</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
              {stepLog.map((action, actionIndex) => (
                <div key={actionIndex} className="border p-2 rounded text-sm">
                  <div>
                    ByteWord {String.fromCharCode(65 + action.sourceIndex)} 
                    ({action.operation}) → 
                    ByteWord {String.fromCharCode(65 + action.targetIndex)}
                  </div>
                  {action.result === 'copy' ? (
                    <div className="text-green-600">
                      Copied ByteWord {String.fromCharCode(65 + action.sourceIndex)} (value: {action.sourceBefore})
                    </div>
                  ) : (
                    <div className="text-blue-600">
                      Transformed {action.targetBefore} → {action.targetAfter}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Main simulation component
const ByteWordSimulator = () => {
  const [byteWords, setByteWords] = useState([
    new ByteWord(0x29), // 0010|1001
    new ByteWord(0x35), // 0011|0101
    new ByteWord(0x11)  // 0001|0001
  ]);
  const [quineSystem, setQuineSystem] = useState(null);
  const [activeIndex, setActiveIndex] = useState(null);
  const [showDetails, setShowDetails] = useState(false);
  const [totalSteps, setTotalSteps] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1000); // 1000ms = 1s

  // Initialize the quine system
  useEffect(() => {
    resetSimulation();
  }, []);

  // Handle automatic playback
  useEffect(() => {
    let interval;
    if (isPlaying) {
      interval = setInterval(() => {
        stepSimulation();
      }, playbackSpeed);
    }
    return () => clearInterval(interval);
  }, [isPlaying, playbackSpeed, quineSystem]);

  const resetSimulation = () => {
    const newSystem = new QuineSystem(byteWords);
    setQuineSystem(newSystem);
    setTotalSteps(0);
    setIsPlaying(false);
  };

  const stepSimulation = () => {
    if (!quineSystem) return;
    quineSystem.step();
    setQuineSystem({ ...quineSystem });
    setTotalSteps(totalSteps + 1);
    
    // Safety check - stop if too many ByteWords
    if (quineSystem.byteWords.length > 100) {
      setIsPlaying(false);
      alert("Simulation stopped: too many ByteWords created (>100)");
    }
  };

  const handleByteWordChange = (index, byteWord) => {
    const newByteWords = [...byteWords];
    newByteWords[index] = byteWord;
    setByteWords(newByteWords);
    resetSimulation();
  };

  const addByteWord = () => {
    // Add a new ByteWord with random value
    const newByteWord = new ByteWord(Math.floor(Math.random() * 256));
    setByteWords([...byteWords, newByteWord]);
    resetSimulation();
  };

  const removeByteWord = (index) => {
    if (byteWords.length <= 1) return;
    const newByteWords = byteWords.filter((_, i) => i !== index);
    setByteWords(newByteWords);
    resetSimulation();
  };

  const togglePlayback = () => {
    setIsPlaying(!isPlaying);
  };

  const renderSimulationControls = () => {
    return (
      <div className="mb-4 p-3 bg-gray-100 rounded flex flex-wrap gap-2 items-center">
        <button 
          onClick={resetSimulation}
          className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
        >
          Reset
        </button>
        <button 
          onClick={stepSimulation}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          disabled={isPlaying}
        >
          Step
        </button>
        <button 
          onClick={togglePlayback}
          className={`px-4 py-2 ${isPlaying ? 'bg-yellow-500 hover:bg-yellow-600' : 'bg-green-500 hover:bg-green-600'} text-white rounded`}
        >
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <div className="flex items-center ml-4">
          <label className="mr-2">Speed:</label>
          <select 
            value={playbackSpeed}
            onChange={(e) => setPlaybackSpeed(parseInt(e.target.value))}
            className="p-1 border rounded"
          >
            <option value="2000">Slow (2s)</option>
            <option value="1000">Normal (1s)</option>
            <option value="500">Fast (0.5s)</option>
            <option value="250">Very Fast (0.25s)</option>
          </select>
        </div>
        <div className="flex items-center ml-4">
          <label className="mr-2">
            <input 
              type="checkbox" 
              checked={showDetails} 
              onChange={() => setShowDetails(!showDetails)}
              className="mr-1"
            />
            Show Details
          </label>
        </div>
        <div className="ml-auto font-medium">
          Steps: {totalSteps} | ByteWords: {quineSystem ? quineSystem.byteWords.length : 0}
        </div>
      </div>
    );
  };

  const renderByteWordEditor = (byteWord, index) => {
    const isActive = activeIndex === index;
    
    return (
      <div 
        key={index} 
        className={`mb-6 p-3 border rounded ${isActive ? 'border-blue-500 ring-2 ring-blue-300' : 'border-gray-200'}`}
      >
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-lg font-semibold">ByteWord {String.fromCharCode(65 + index)}</h3>
          <button 
            onClick={() => removeByteWord(index)}
            className="p-1 text-red-500 hover:text-red-700"
            title="Remove ByteWord"
          >
            ×
          </button>
        </div>
        <div 
          onClick={() => setActiveIndex(isActive ? null : index)}
          className="cursor-pointer mb-2"
        >
          <ByteWordDisplay byteWord={byteWord} index={index} showDetails={showDetails} />
        </div>
        {isActive && (
          <ByteWordEditor byteWord={byteWord} onChange={(newByteWord) => handleByteWordChange(index, newByteWord)} />
        )}
      </div>
    );
  };

  const renderCurrentState = () => {
    if (!quineSystem) return null;
    
    return (
      <div className="mt-6">
        <h3 className="text-xl font-semibold mb-3">Current State (Step {totalSteps})</h3>
        <div className="flex flex-wrap gap-2">
          {quineSystem.byteWords.map((byteWord, index) => (
            <ByteWordDisplay 
              key={index} 
              byteWord={byteWord} 
              index={index} 
              showDetails={showDetails}
            />
          ))}
        </div>
      </div>
    );
  };

  const renderVisualization = () => {
    if (!quineSystem) return null;
    
    return (
      <div className="mt-6">
        <h3 className="text-xl font-semibold mb-3">Visualization</h3>
        {quineSystem.replicationHistory.length > 1 && (
          <TransformationVisualizer quineSystem={quineSystem} />
        )}
      </div>
    );
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">ByteWord Self-Replicating System</h1>
      
      {renderSimulationControls()}
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h2 className="text-xl font-semibold mb-3">Initial Configuration</h2>
          {byteWords.map((byteWord, index) => renderByteWordEditor(byteWord, index))}
          <button 
            onClick={addByteWord}
            className="w-full py-2 bg-green-500 text-white rounded hover:bg-green-600"
          >
            + Add ByteWord
          </button>
        </div>
        
        <div>
          {renderCurrentState()}
          {renderVisualization()}
        </div>
      </div>
      
      <div className="mt-8">
        <h2 className="text-xl font-semibold mb-3">System Analysis</h2>
        {quineSystem && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-gray-100 rounded">
              <h3 className="text-lg font-medium mb-2">Growth Analysis</h3>
              <div>
                <p>Initial ByteWords: {quineSystem.replicationHistory[0].length}</p>
                <p>Current ByteWords: {quineSystem.byteWords.length}</p>
                <p>Growth Rate: {quineSystem.byteWords.length - quineSystem.replicationHistory[0].length} new ByteWords after {totalSteps} steps</p>
              </div>
            </div>
            
            <div className="p-4 bg-gray-100 rounded">
              <h3 className="text-lg font-medium mb-2">Morphism Distribution</h3>
              <div>
                {[0, 1, 2, 3, 4, 5, 6, 7].map(morphism => {
                  const count = quineSystem.byteWords.filter(bw => bw.morphism === morphism).length;
                  return (
                    <div key={morphism} className="flex justify-between">
                      <span>{getMorphismDescription(morphism)}:</span>
                      <span>{count} ({Math.round(count / quineSystem.byteWords.length * 100)}%)</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ByteWordSimulator;