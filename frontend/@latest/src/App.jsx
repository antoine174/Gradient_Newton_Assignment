import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import './App.css'; // Add some basic styles here later

export default function App() {
  const [method, setMethod] = useState('gd');
  const [alpha, setAlpha] = useState(0.1);
  const [a0, setA0] = useState(0);
  const [a1, setA1] = useState(0);
  const [maxIter, setMaxIter] = useState(1000);
  
  const [data, setData] = useState({ summary: {}, history: [] });
  const [loading, setLoading] = useState(false);

  const runOptimization = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `http://127.0.0.1:8000/api/run-optimization?method=${method}&alpha=${alpha}&a0_init=${a0}&a1_init=${a1}&max_iter=${maxIter}`
      );
      const data = await response.json();
      
      // Safety check if history is empty (diverged immediately)
      const hasHistory = data.history && data.history.length > 0;
      const finalHistory = hasHistory ? data.history[data.history.length - 1] : { loss: null, a0: 0, a1: 0 };
      
      const summary = {
        status: data.status,
        total_iterations: hasHistory ? data.history.length : 0,
        final_loss: finalHistory.loss,
        final_weights: {
          a0: finalHistory.a0,
          a1: finalHistory.a1
        }
      };

      setData({ history: data.history || [], summary: summary });
    } catch (error) {
      console.error("Error fetching data:", error);
    }
    setLoading(false);
  };

  useEffect(() => {
    runOptimization();
  }, []);

  return (
    <div className="app-container">
      
      <div className="sidebar">
        <div className="sidebar-header">
          <h2>Optimization</h2>
          <p>Housing Price Regression</p>
        </div>
        
        <div className="control-group">
          <label>Algorithm</label>
          <select value={method} onChange={(e) => setMethod(e.target.value)} className="styled-select">
            <option value="gd">Gradient Descent</option>
            <option value="newton">Newton's Method</option>
          </select>
        </div>

        <div className="control-group">
          <label>Learning Rate</label>
          <input type="number" min="0.0001" step="0.001" value={alpha} onChange={(e) => setAlpha(parseFloat(e.target.value))} className="styled-input" />
        </div>

        <div className="control-group">
          <label>Initial a0 (Intercept)</label>
          <input type="number" value={a0} onChange={(e) => setA0(parseFloat(e.target.value))} className="styled-input" />
        </div>

        <div className="control-group">
          <label>Initial a1 (Slope)</label>
          <input type="number" value={a1} onChange={(e) => setA1(parseFloat(e.target.value))} className="styled-input" />
        </div>

        <div className="control-group">
          <label>Max Iterations</label>
          <input type="number" value={maxIter} onChange={(e) => setMaxIter(parseInt(e.target.value))} className="styled-input" />
        </div>

        <button onClick={runOptimization} disabled={loading} className="btn-primary">
          {loading ? 'Running...' : 'Run Optimization'}
        </button>

        {data.summary.status && (
          <div className="results-card">
            <h3>Summary</h3>
            <div className="result-item" style={{ marginBottom: '8px' }}>
              <span>Status</span>
              <span style={{ 
                color: data.summary.status.includes('Optimal') ? '#10b981' : 
                       data.summary.status.includes('Diverged') ? '#ef4444' : '#f59e0b',
                fontWeight: 700 
              }}>
                {data.summary.status}
              </span>
            </div>
            <div className="result-item">
              <span>Iterations</span>
              <span>{data.summary.total_iterations}</span>
            </div>
            {data.summary.final_loss !== null && (
              <>
                <div className="result-item">
                  <span>Final Loss (MSE)</span>
                  <span>{data.summary.final_loss.toFixed(4)}</span>
                </div>
                <div className="result-item">
                  <span>a0 Weight</span>
                  <span>{data.summary.final_weights.a0.toFixed(4)}</span>
                </div>
                <div className="result-item">
                  <span>a1 Weight</span>
                  <span>{data.summary.final_weights.a1.toFixed(4)}</span>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      <div className="main-content">
        
        <div className="chart-card">
          <div className="chart-header">
            <h3 className="chart-title">Loss Curve</h3>
            <p className="chart-subtitle">Cost function value over optimization iterations</p>
          </div>
          <div style={{ width: '100%', height: '400px', minHeight: '400px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.history}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} vertical={false} />
                <XAxis dataKey="iter" label={{ value: 'Iterations', position: 'insideBottomRight', offset: -5 }} axisLine={false} tickLine={false} />
                <YAxis scale="log" domain={['auto', 'auto']} label={{ value: 'Log(Loss)', angle: -90, position: 'insideLeft' }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                <Line type="monotone" dataKey="loss" stroke="#6366f1" strokeWidth={3} dot={false} activeDot={{ r: 6, fill: '#6366f1', stroke: '#fff', strokeWidth: 2 }} />
              </LineChart>
            </ResponsiveContainer>
        </div>
        </div>

        <div className="chart-card">
          <div className="chart-header">
            <h3 className="chart-title">Weight Trajectory</h3>
            <p className="chart-subtitle">Path taken by intercept ($a_0$) and slope ($a_1$)</p>
          </div>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} vertical={false} />
                <XAxis dataKey="a0" type="number" name="Weight a0" label={{ value: 'a0 (Intercept)', position: 'insideBottomRight', offset: -5 }} axisLine={false} tickLine={false} />
                <YAxis dataKey="a1" type="number" name="Weight a1" label={{ value: 'a1 (Slope)', angle: -90, position: 'insideLeft' }} axisLine={false} tickLine={false} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                <Scatter name="Path" data={data.history} fill="#a855f7" line shape="circle" lineType="joint" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>
    </div>
  );
}