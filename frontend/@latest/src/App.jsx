import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import './App.css'; // Add some basic styles here later

export default function App() {
  const [method, setMethod] = useState('gd');
  const [alpha, setAlpha] = useState(0.1);
  const [a0, setA0] = useState(0);
  const [a1, setA1] = useState(0);
  
  const [data, setData] = useState({ summary: {}, history: [] });
  const [loading, setLoading] = useState(false);

  const runOptimization = async () => {
    setLoading(true);
    try {
      const response = await axios.get('http://localhost:8000/api/run-optimization', {
        params: { method, alpha, a0_init: a0, a1_init: a1 }
      });
      setData(response.data);
    } catch (error) {
      console.error("Error fetching data:", error);
    }
    setLoading(false);
  };

  useEffect(() => {
    runOptimization();
  }, []);

  return (
    <div style={{ display: 'flex', minHeight: '100vh', fontFamily: 'system-ui, sans-serif', backgroundColor: '#f4f4f9' }}>
      
      <div style={{ width: '300px', padding: '20px', backgroundColor: '#fff', boxShadow: '2px 0 5px rgba(0,0,0,0.1)' }}>
        <h2 style={{ color: '#333' }}>Optimization Control</h2>
        
        <div style={{ marginBottom: '20px' }}>
          <label><strong>Algorithm:</strong></label>
          <select value={method} onChange={(e) => setMethod(e.target.value)} style={{ width: '100%', padding: '8px', marginTop: '5px' }}>
            <option value="gd">Gradient Descent</option>
            <option value="newton">Newton's Method</option>
          </select>
        </div>

        <div style={{ marginBottom: '20px' }}>
          <label><strong>Learning Rate ($\alpha$): {alpha}</strong></label>
          <input type="range" min="0.001" max="1" step="0.01" value={alpha} onChange={(e) => setAlpha(parseFloat(e.target.value))} style={{ width: '100%' }} />
        </div>

        <div style={{ marginBottom: '20px' }}>
          <label><strong>Initial $a_0$:</strong></label>
          <input type="number" value={a0} onChange={(e) => setA0(parseFloat(e.target.value))} style={{ width: '100%', padding: '8px', marginTop: '5px' }} />
        </div>

        <div style={{ marginBottom: '20px' }}>
          <label><strong>Initial $a_1$:</strong></label>
          <input type="number" value={a1} onChange={(e) => setA1(parseFloat(e.target.value))} style={{ width: '100%', padding: '8px', marginTop: '5px' }} />
        </div>

        <button 
          onClick={runOptimization} 
          style={{ width: '100%', padding: '10px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer', fontWeight: 'bold' }}
        >
          {loading ? 'Running...' : 'Run Optimization'}
        </button>

        {data.summary.final_loss && (
          <div style={{ marginTop: '30px', padding: '15px', backgroundColor: '#e9ecef', borderRadius: '8px' }}>
            <h3>Results</h3>
            <p><strong>Iterations:</strong> {data.summary.total_iterations}</p>
            <p><strong>Final Loss:</strong> {data.summary.final_loss.toFixed(4)}</p>
            <p><strong>$a_0$:</strong> {data.summary.final_weights.a0.toFixed(4)}</p>
            <p><strong>$a_1$:</strong> {data.summary.final_weights.a1.toFixed(4)}</p>
          </div>
        )}
      </div>

      <div style={{ flex: 1, padding: '20px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        
        <div style={{ flex: 1, backgroundColor: '#fff', padding: '20px', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
          <h3>Loss Curve (Iterations vs SSE)</h3>
          <ResponsiveContainer width="100%" height="90%">
            <LineChart data={data.history}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.5} />
              <XAxis dataKey="iter" label={{ value: 'Iterations', position: 'insideBottomRight', offset: -5 }} />
              <YAxis scale="log" domain={['auto', 'auto']} label={{ value: 'Loss (Log Scale)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Line type="monotone" dataKey="loss" stroke="#ff7300" strokeWidth={3} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{ flex: 1, backgroundColor: '#fff', padding: '20px', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
          <h3>Weight Trajectory ($a_0$ vs $a_1$)</h3>
          <ResponsiveContainer width="100%" height="90%">
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" opacity={0.5} />
              <XAxis dataKey="a0" type="number" name="Weight a0" label={{ value: 'a0', position: 'insideBottomRight', offset: -5 }} />
              <YAxis dataKey="a1" type="number" name="Weight a1" label={{ value: 'a1', angle: -90, position: 'insideLeft' }} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Scatter name="Path" data={data.history} fill="#8884d8" line shape="circle" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

      </div>
    </div>
  );
}