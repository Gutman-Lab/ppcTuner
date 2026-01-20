import { useState } from 'react'
import { safeJsonParse } from '../utils/api'
import type { RoiHueSampleReport } from '../types'

interface PpcParametersProps {
  itemId: string | null
  hueValue: number
  hueWidth: number
  saturationMinimum: number
  intensityUpperLimit: number
  intensityWeakThreshold: number
  intensityStrongThreshold: number
  intensityLowerLimit: number
  setHueValue: (value: number) => void
  setHueWidth: (value: number) => void
  setSaturationMinimum: (value: number) => void
  setIntensityUpperLimit: (value: number) => void
  setIntensityWeakThreshold: (value: number) => void
  setIntensityStrongThreshold: (value: number) => void
  setIntensityLowerLimit: (value: number) => void
  roiHueReport: RoiHueSampleReport | null
  setRoiHueReport: (report: RoiHueSampleReport | null) => void
  onAfterRoiSample?: () => void
  roiSamplingMode: 'dab_biased' | 'stratified'
  setRoiSamplingMode: (mode: 'dab_biased' | 'stratified') => void
}

export function PpcParameters({
  itemId,
  hueValue,
  hueWidth,
  saturationMinimum: _saturationMinimum,
  intensityUpperLimit: _intensityUpperLimit,
  intensityWeakThreshold,
  intensityStrongThreshold,
  intensityLowerLimit: _intensityLowerLimit,
  setHueValue,
  setHueWidth,
  setSaturationMinimum,
  setIntensityUpperLimit,
  setIntensityWeakThreshold,
  setIntensityStrongThreshold,
  setIntensityLowerLimit,
  roiHueReport,
  setRoiHueReport,
  onAfterRoiSample,
  roiSamplingMode,
  setRoiSamplingMode,
}: PpcParametersProps) {
  const [showTooltip, setShowTooltip] = useState(false)
  const [roiHueLoading, setRoiHueLoading] = useState(false)
  
  return (
    <div style={{ marginBottom: '1rem', padding: '0.75rem', backgroundColor: '#f9f9f9', borderRadius: '4px', fontSize: '0.875rem' }}>
      <div style={{ marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
          <label style={{ color: '#333', fontWeight: '500' }}>Hue Value:</label>
          <span style={{ color: '#666', fontSize: '0.8rem' }}>{(hueValue * 360).toFixed(1)}° ({hueValue.toFixed(3)})</span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.001"
          value={hueValue}
          onChange={(e) => setHueValue(parseFloat(e.target.value))}
          style={{ width: '100%' }}
        />
      </div>
      <div style={{ marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
          <label style={{ color: '#333', fontWeight: '500' }}>Hue Width:</label>
          <span style={{ color: '#666', fontSize: '0.8rem' }}>{(hueWidth * 360).toFixed(1)}° ({hueWidth.toFixed(3)})</span>
        </div>
        <input
          type="range"
          min="0.01"
          max="0.5"
          step="0.01"
          value={hueWidth}
          onChange={(e) => setHueWidth(parseFloat(e.target.value))}
          style={{ width: '100%' }}
        />
      </div>
      
      {/* Intensity Threshold Controls */}
      <div style={{ marginTop: '1rem', paddingTop: '0.75rem', borderTop: '1px solid #ddd' }}>
        <div style={{ marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ fontWeight: 'bold', color: '#333' }}>Staining Intensity Thresholds:</span>
          <div style={{ position: 'relative', display: 'inline-block' }}>
            <button
              onClick={() => setShowTooltip(!showTooltip)}
              onMouseEnter={() => setShowTooltip(true)}
              onMouseLeave={() => setShowTooltip(false)}
              style={{
                width: '20px',
                height: '20px',
                borderRadius: '50%',
                border: '1px solid #666',
                backgroundColor: '#fff',
                color: '#666',
                fontSize: '0.75rem',
                fontWeight: 'bold',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: 0,
                lineHeight: 1
              }}
              title="Click or hover for HSI intensity explanation"
            >
              ?
            </button>
            {showTooltip && (
              <div
                style={{
                  position: 'absolute',
                  top: '25px',
                  left: '0',
                  zIndex: 1000,
                  width: '280px',
                  padding: '0.75rem',
                  backgroundColor: '#fff3cd',
                  border: '1px solid #ffc107',
                  borderRadius: '4px',
                  fontSize: '0.75rem',
                  color: '#666',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
                  lineHeight: '1.5'
                }}
                onMouseEnter={() => setShowTooltip(true)}
                onMouseLeave={() => setShowTooltip(false)}
              >
                <strong>⚠️ Important:</strong> In HSI color space, "intensity" = pixel brightness (0=dark, 1=bright).<br />
                <strong>Strong staining</strong> = dark brown DAB = <strong>LOW pixel intensity</strong> (closer to 0)<br />
                <strong>Weak staining</strong> = light brown DAB = <strong>HIGH pixel intensity</strong> (closer to 1)
              </div>
            )}
          </div>
        </div>
        <div style={{ marginBottom: '0.75rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
            <label style={{ color: '#333', fontWeight: '500' }}>Weak Staining (Light Brown, ≥):</label>
            <span style={{ color: '#666', fontSize: '0.8rem' }}>{intensityWeakThreshold.toFixed(3)}</span>
          </div>
          <input
            type="range"
            min={Math.max(0.1, intensityStrongThreshold + 0.05)}
            max="1"
            step="0.01"
            value={intensityWeakThreshold}
            onChange={(e) => {
              const newValue = parseFloat(e.target.value)
              setIntensityWeakThreshold(newValue)
              // Ensure strong threshold is always less than weak threshold
              if (newValue <= intensityStrongThreshold) {
                setIntensityStrongThreshold(Math.max(0.05, newValue - 0.05))
              }
            }}
            style={{ width: '100%' }}
          />
          <div style={{ fontSize: '0.75rem', color: '#999', marginTop: '0.25rem' }}>
            Pixels with intensity ≥ {intensityWeakThreshold.toFixed(2)} = <strong>weak</strong> (light brown, bright pixels)
          </div>
        </div>
        <div style={{ marginBottom: '0.75rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
            <label style={{ color: '#333', fontWeight: '500' }}>Strong Staining (Dark Brown, &lt;):</label>
            <span style={{ color: '#666', fontSize: '0.8rem' }}>{intensityStrongThreshold.toFixed(3)}</span>
          </div>
          <input
            type="range"
            min="0.05"
            max={Math.min(0.95, intensityWeakThreshold - 0.05)}
            step="0.01"
            value={intensityStrongThreshold}
            onChange={(e) => {
              const newValue = parseFloat(e.target.value)
              setIntensityStrongThreshold(newValue)
              // Ensure weak threshold is always greater than strong threshold
              if (newValue >= intensityWeakThreshold) {
                setIntensityWeakThreshold(Math.min(1.0, newValue + 0.05))
              }
            }}
            style={{ width: '100%' }}
          />
          <div style={{ fontSize: '0.75rem', color: '#999', marginTop: '0.25rem' }}>
            Pixels with intensity &lt; {intensityStrongThreshold.toFixed(2)} = <strong>strong</strong> (dark brown, dark pixels)
          </div>
        </div>
        <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.5rem', padding: '0.5rem', backgroundColor: '#f0f8ff', borderRadius: '4px' }}>
          <strong>Plain (medium) staining:</strong> {intensityStrongThreshold.toFixed(2)} ≤ intensity &lt; {intensityWeakThreshold.toFixed(2)}<br />
          <span style={{ fontSize: '0.7rem', fontStyle: 'italic', color: '#555' }}>
            💡 Remember: Lower intensity value = darker brown = stronger staining. Higher intensity value = lighter brown = weaker staining.
          </span>
        </div>
      </div>
      {/* Visual Hue Range Indicator */}
      <div style={{ marginBottom: '0.75rem', padding: '0.5rem', backgroundColor: '#fff', borderRadius: '4px', border: '1px solid #ddd' }}>
        <div style={{ fontSize: '0.75rem', color: '#666', marginBottom: '0.25rem' }}>Hue Range:</div>
        <div style={{ 
          height: '30px', 
          borderRadius: '4px',
          background: `linear-gradient(to right, 
            hsl(${(hueValue - hueWidth/2) * 360}, 70%, 50%),
            hsl(${hueValue * 360}, 70%, 50%),
            hsl(${(hueValue + hueWidth/2) * 360}, 70%, 50%))`,
          border: '1px solid #ccc',
          position: 'relative'
        }}>
          <div style={{
            position: 'absolute',
            left: '50%',
            top: '50%',
            transform: 'translate(-50%, -50%)',
            width: '2px',
            height: '100%',
            backgroundColor: '#000',
            opacity: 0.5
          }} />
        </div>
        <div style={{ fontSize: '0.7rem', color: '#999', marginTop: '0.25rem', textAlign: 'center' }}>
          {(hueValue - hueWidth/2) * 360}° - {(hueValue + hueWidth/2) * 360}°
        </div>
      </div>
      <button
        onClick={async () => {
          if (!itemId) return
          try {
            const response = await fetch(`/api/ppc/auto-detect-hue?item_id=${itemId}&thumbnail_width=1024`)
            if (response.ok) {
              const data = await safeJsonParse(response)
              setHueValue(data.hue_value)
              setHueWidth(data.hue_width)
              setSaturationMinimum(data.saturation_minimum)
              setIntensityUpperLimit(data.intensity_upper_limit)
              setIntensityWeakThreshold(data.intensity_weak_threshold)
              setIntensityStrongThreshold(data.intensity_strong_threshold)
              setIntensityLowerLimit(data.intensity_lower_limit)
            } else {
              console.error('Auto-detect failed:', response.status)
            }
          } catch (error) {
            console.error('Error auto-detecting hue:', error)
          }
        }}
        disabled={!itemId}
        style={{
          padding: '0.4rem 0.8rem',
          fontSize: '0.8rem',
          backgroundColor: '#2196F3',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: !itemId ? 'not-allowed' : 'pointer',
          opacity: !itemId ? 0.6 : 1,
          width: '100%'
        }}
      >
        Auto-detect Hue Parameters
      </button>

      <button
        onClick={async () => {
          if (!itemId) return
          setRoiHueLoading(true)
          try {
            const params = new URLSearchParams({
              item_id: itemId,
              n_rois: '5',
              roi_fraction: '0.08',
              roi_output_width: '1024',
              min_tissue_fraction: '0.6',
              min_dab_band_fraction: '0.01',
              min_roi_separation: '0.072',
              sampling_mode: roiSamplingMode,
              max_candidates: '10',
              thumbnail_width: '1024',
            })
            const response = await fetch(`/api/ppc/auto-detect-hue-sampled?${params.toString()}`)
            if (response.ok) {
              const data = await safeJsonParse(response)
              setRoiHueReport(data as RoiHueSampleReport)
              onAfterRoiSample?.()
            } else {
              const txt = await response.text()
              console.error('ROI hue sampling failed:', response.status, txt)
              alert(`ROI hue sampling failed: ${response.status}`)
            }
          } catch (error) {
            console.error('Error sampling ROI hue:', error)
            alert(`Error: ${error}`)
          } finally {
            setRoiHueLoading(false)
          }
        }}
        disabled={!itemId || roiHueLoading}
        style={{
          marginTop: '0.5rem',
          padding: '0.4rem 0.8rem',
          fontSize: '0.8rem',
          backgroundColor: roiHueLoading ? '#ccc' : '#6c5ce7',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: (!itemId || roiHueLoading) ? 'not-allowed' : 'pointer',
          opacity: (!itemId || roiHueLoading) ? 0.6 : 1,
          width: '100%'
        }}
      >
        {roiHueLoading ? 'Sampling ROIs...' : 'Sample 5 ROIs (≈10x) + Compare'}
      </button>

      <div style={{ marginTop: '0.5rem', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
        <label style={{ fontSize: '0.75rem', color: '#555', fontWeight: 600 }}>ROI sampling mode:</label>
        <select
          value={roiSamplingMode}
          onChange={(e) => setRoiSamplingMode(e.target.value as 'dab_biased' | 'stratified')}
          style={{
            fontSize: '0.8rem',
            padding: '0.25rem 0.4rem',
            borderRadius: '6px',
            border: '1px solid #ddd',
            background: '#fff',
          }}
          title="Choose how the backend picks ROI locations"
        >
          <option value="dab_biased">DAB-biased (current)</option>
          <option value="stratified">Stratified (spread across tissue)</option>
        </select>
      </div>

      {roiHueReport?.summary && (
        <div style={{ marginTop: '0.75rem', padding: '0.75rem', border: '1px solid #ddd', borderRadius: '6px', backgroundColor: '#fff' }}>
          <div style={{ fontWeight: 700, marginBottom: '0.5rem', color: '#333' }}>ROI Hue Stability</div>
          {(!roiHueReport.debug_rois || roiHueReport.debug_rois.length === 0) ? (
            <div style={{ marginBottom: '0.5rem', fontSize: '0.75rem', color: '#a15c00' }}>
              Note: no <code>debug_rois</code> returned, so the viewer can only draw accepted ROIs. If you expected BAD boxes too, restart the backend and re-run sampling.
            </div>
          ) : null}
          {roiHueReport.summary.likely_negative ? (
            <div style={{ marginBottom: '0.5rem', padding: '0.5rem', backgroundColor: '#fff3cd', border: '1px solid #ffc107', borderRadius: '4px', color: '#6b5b00' }}>
              <strong>Likely DAB-negative slide</strong> (baseline DAB-band frac≈{(roiHueReport.summary.baseline_dab_band_fraction ?? 0).toFixed?.(4) ?? roiHueReport.summary.baseline_dab_band_fraction}).<br />
              ROI sampling will return <strong>debug candidates</strong> without doing expensive region fetches.
            </div>
          ) : null}
          <div style={{ fontSize: '0.8rem', color: '#333', lineHeight: 1.5 }}>
            <div style={{ fontSize: '0.75rem', color: '#666' }}>
              <strong>Mode</strong>: {roiHueReport.summary.sampling_mode ?? 'unknown'}{' '}
              | <strong>debug_rois</strong>: {roiHueReport.debug_rois?.length ?? 0}
              {typeof roiHueReport.summary.max_candidates !== 'undefined' ? (
                <> | <strong>max_candidates</strong>: {roiHueReport.summary.max_candidates}</>
              ) : null}
            </div>
            {(typeof roiHueReport.summary.rejected_tissue !== 'undefined' ||
              typeof roiHueReport.summary.rejected_dab !== 'undefined' ||
              typeof roiHueReport.summary.rejected_too_close !== 'undefined' ||
              typeof roiHueReport.summary.fetch_failures !== 'undefined') ? (
              <div style={{ fontSize: '0.75rem', color: '#666' }}>
                <strong>rejects</strong>:
                {' '}tissue={roiHueReport.summary.rejected_tissue ?? 0},
                {' '}strong_dab={roiHueReport.summary.rejected_dab ?? 0},
                {' '}close={roiHueReport.summary.rejected_too_close ?? 0},
                {' '}fetch={roiHueReport.summary.fetch_failures ?? 0}
              </div>
            ) : null}
            <div>
              <strong>Baseline (thumbnail)</strong>: H={roiHueReport.summary.baseline_hue_value} ({(roiHueReport.summary.baseline_hue_value * 360).toFixed(1)}°),
              W={roiHueReport.summary.baseline_hue_width} ({(roiHueReport.summary.baseline_hue_width * 360).toFixed(1)}°)
            </div>
            <div>
              <strong>ROIs</strong>: n={roiHueReport.summary.n_rois_computed}/{roiHueReport.summary.n_rois_requested},{' '}
              frac={roiHueReport.summary.roi_fraction}, outW={roiHueReport.summary.roi_output_width}
            </div>
            <div>
              <strong>Hue mean±std</strong>: {roiHueReport.summary.hue_value_mean?.toFixed?.(4) ?? roiHueReport.summary.hue_value_mean}
              {' '}±{' '}
              {roiHueReport.summary.hue_value_std?.toFixed?.(4) ?? roiHueReport.summary.hue_value_std}
            </div>
            <div>
              <strong>Width mean±std</strong>: {roiHueReport.summary.hue_width_mean?.toFixed?.(4) ?? roiHueReport.summary.hue_width_mean}
              {' '}±{' '}
              {roiHueReport.summary.hue_width_std?.toFixed?.(4) ?? roiHueReport.summary.hue_width_std}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
