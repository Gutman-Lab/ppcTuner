import { useEffect, useState } from 'react'
import type { Dispatch, SetStateAction } from 'react'
import { SlideViewerSection } from './SlideViewerSection'
import { HistogramSection } from './HistogramSection'
import { ParameterSetsPanel } from './ParameterSetsPanel'
import { PpcControls } from './PpcControls'
import { PpcParameters } from './PpcParameters'
import { ViewportInfo } from './ViewportInfo'
import { CapturedRegionPanel } from './CapturedRegionPanel'
import { PpcResults } from './PpcResults'
import { AutoComputeControls } from './AutoComputeControls'
import { useLocalStorage } from '../hooks/useLocalStorage'
import type { SavedParameterSet, PpcResult, CapturedRegion, RoiHueSampleReport } from '../types'
import type { Item } from 'bdsa-react-components'

interface ImageViewerSectionProps {
  selectedItem: Item
  itemId: string
  apiBaseUrl: string
  dsaToken: string | null
  // Histogram
  histogramData: any
  histogramLoading: boolean
  // Image dimensions and viewport
  imageDimensions: { width: number, height: number } | null
  currentViewport: { x: number, y: number, width: number, height: number, zoom: number } | null
  setCurrentViewport: (viewport: { x: number, y: number, width: number, height: number, zoom: number } | null) => void
  // Region capture (multiple)
  capturedRegions: CapturedRegion[]
  setCapturedRegions: Dispatch<SetStateAction<CapturedRegion[]>>
  // PPC
  ppcData: PpcResult | null
  ppcLoading: boolean
  intensityMap: any
  intensityMapLoading: boolean
  computePpc: (itemId: string) => void
  // HSI Parameters
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
  // PPC Label Overlay
  showPpcLabel: boolean
  setShowPpcLabel: (show: boolean) => void
  ppcLabelOpacity: number
  setPpcLabelOpacity: (opacity: number) => void
  showWeak: boolean
  showPlain: boolean
  showStrong: boolean
  setShowWeak: (show: boolean) => void
  setShowPlain: (show: boolean) => void
  setShowStrong: (show: boolean) => void
  labelColorScheme: string
  setLabelColorScheme: (scheme: string) => void
  // Tissue mask
  showTissueMask: boolean
  setShowTissueMask: (show: boolean) => void
  // Auto compute
  autoCompute: boolean
  setAutoCompute: (enabled: boolean) => void
  // Parameter sets
  savedParameterSets: SavedParameterSet[]
  setSavedParameterSets: Dispatch<SetStateAction<SavedParameterSet[]>>
  nextSetId: number
  setNextSetId: Dispatch<SetStateAction<number>>
  // Auto-detect hue
  autoDetectHue: (itemId: string) => Promise<void>
}

export function ImageViewerSection({
  selectedItem,
  itemId,
  apiBaseUrl,
  dsaToken,
  histogramData,
  histogramLoading,
  imageDimensions,
  currentViewport,
  setCurrentViewport,
  capturedRegions,
  setCapturedRegions,
  ppcData,
  ppcLoading,
  intensityMap,
  intensityMapLoading,
  computePpc,
  hueValue,
  hueWidth,
  saturationMinimum,
  intensityUpperLimit,
  intensityWeakThreshold,
  intensityStrongThreshold,
  intensityLowerLimit,
  setHueValue,
  setHueWidth,
  setSaturationMinimum,
  setIntensityUpperLimit,
  setIntensityWeakThreshold,
  setIntensityStrongThreshold,
  setIntensityLowerLimit,
  showPpcLabel,
  setShowPpcLabel,
  ppcLabelOpacity,
  setPpcLabelOpacity,
  showWeak,
  showPlain,
  showStrong,
  setShowWeak,
  setShowPlain,
  setShowStrong,
  labelColorScheme,
  setLabelColorScheme,
  showTissueMask,
  setShowTissueMask,
  autoCompute,
  setAutoCompute,
  savedParameterSets,
  setSavedParameterSets,
  nextSetId,
  setNextSetId,
}: ImageViewerSectionProps) {
  const [isRightSidebarCollapsed, setIsRightSidebarCollapsed] = useLocalStorage<boolean>(
    'ppcTuner_rightSidebarCollapsed',
    true
  )
  const [isPpcPanelCollapsed, setIsPpcPanelCollapsed] = useLocalStorage<boolean>(
    'ppcTuner_ppcPanelCollapsed',
    false
  )

  // ROI sampling report + optional viewer overlay of sampled squares
  const [roiHueReport, setRoiHueReport] = useState<RoiHueSampleReport | null>(null)
  const [showSampledRois, setShowSampledRois] = useState(false)
  const [roiSamplingMode, setRoiSamplingMode] = useLocalStorage<'dab_biased' | 'stratified'>(
    'ppcTuner_roiSamplingMode',
    'dab_biased'
  )

  // Reset ROI sampling state when the slide changes (sampling is slide-specific).
  useEffect(() => {
    setRoiHueReport(null)
    setShowSampledRois(false)
  }, [itemId])
  
  // Automatically enable PPC label overlay when PPC data becomes available
  // This ensures the overlay appears after computing PPC (replacing the old thumbnail panel behavior)
  useEffect(() => {
    if (ppcData && ppcData.method === 'hsi' && !showPpcLabel) {
      setShowPpcLabel(true)
    }
  }, [ppcData, showPpcLabel, setShowPpcLabel])

  const lastCapturedRegion = capturedRegions.length > 0 ? capturedRegions[capturedRegions.length - 1].region : null
  
  return (
    <div className="image-viewer-section">
      <h2>Selected Image</h2>
      <div style={{ 
        display: 'flex',
        gap: '1.5rem',
        alignItems: 'flex-start',
        width: '100%',
        margin: '0 auto'
      }}>
        {/* Image in the middle */}
        <div style={{ 
          flex: '1 1 60%',
          border: '1px solid #ddd', 
          borderRadius: '8px', 
          padding: '1rem',
          backgroundColor: '#fff',
          textAlign: 'center'
        }}>
          <div style={{ marginBottom: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', alignItems: 'center' }}>
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
              <label style={{ fontSize: '0.875rem', color: '#333', display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={showTissueMask}
                  onChange={(e) => setShowTissueMask(e.target.checked)}
                  style={{ cursor: 'pointer' }}
                />
                Show Tissue Mask
              </label>
              {(roiHueReport?.rois?.length || roiHueReport?.debug_rois?.length) ? (
                <label style={{ fontSize: '0.875rem', color: '#333', display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={showSampledRois}
                    onChange={(e) => setShowSampledRois(e.target.checked)}
                    style={{ cursor: 'pointer' }}
                  />
                  Show Sampled ROIs
                </label>
              ) : null}
              {ppcData?.method === 'hsi' && (
                <label style={{ fontSize: '0.875rem', color: '#333', display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={showPpcLabel}
                    onChange={(e) => setShowPpcLabel(e.target.checked)}
                    style={{ cursor: 'pointer' }}
                  />
                  Show PPC Labels
                </label>
              )}
            </div>
            {showPpcLabel && ppcData?.method === 'hsi' && (
              <PpcControls
                showPpcLabel={showPpcLabel}
                ppcLabelOpacity={ppcLabelOpacity}
                setPpcLabelOpacity={setPpcLabelOpacity}
                showWeak={showWeak}
                showPlain={showPlain}
                showStrong={showStrong}
                setShowWeak={setShowWeak}
                setShowPlain={setShowPlain}
                setShowStrong={setShowStrong}
                labelColorScheme={labelColorScheme}
                setLabelColorScheme={setLabelColorScheme}
              />
            )}
          </div>
          <SlideViewerSection
            itemId={itemId}
            apiBaseUrl={apiBaseUrl}
            dsaToken={dsaToken}
            height="600px"
            currentViewport={currentViewport}
            setCurrentViewport={setCurrentViewport}
            showPpcLabel={showPpcLabel}
            setShowPpcLabel={setShowPpcLabel}
            ppcData={ppcData}
            ppcLabelOpacity={ppcLabelOpacity}
            showWeak={showWeak}
            showPlain={showPlain}
            showStrong={showStrong}
            labelColorScheme={labelColorScheme}
            capturedRegions={capturedRegions}
            setCapturedRegions={setCapturedRegions}
            savedParameterSets={savedParameterSets}
            setSavedParameterSets={setSavedParameterSets}
            hueValue={hueValue}
            hueWidth={hueWidth}
            saturationMinimum={saturationMinimum}
            intensityUpperLimit={intensityUpperLimit}
            intensityWeakThreshold={intensityWeakThreshold}
            intensityStrongThreshold={intensityStrongThreshold}
            intensityLowerLimit={intensityLowerLimit}
            showTissueMask={showTissueMask}
            setShowTissueMask={setShowTissueMask}
            histogramData={histogramData}
            sampledRois={roiHueReport?.rois ?? []}
            sampledRoiDebug={roiHueReport?.debug_rois ?? []}
            showSampledRois={showSampledRois}
            setShowSampledRois={setShowSampledRois}
          />

          {/* Parameter Sets - moved below viewer to free vertical space above */}
          <div style={{ marginTop: '1rem' }}>
            <ParameterSetsPanel
              savedParameterSets={savedParameterSets}
              setSavedParameterSets={setSavedParameterSets}
              nextSetId={nextSetId}
              setNextSetId={setNextSetId}
              ppcData={ppcData}
              itemId={itemId}
              itemName={selectedItem?.name}
              currentViewport={currentViewport}
              hueValue={hueValue}
              hueWidth={hueWidth}
              saturationMinimum={saturationMinimum}
              intensityUpperLimit={intensityUpperLimit}
              intensityWeakThreshold={intensityWeakThreshold}
              intensityStrongThreshold={intensityStrongThreshold}
              intensityLowerLimit={intensityLowerLimit}
              setHueValue={setHueValue}
              setHueWidth={setHueWidth}
              setSaturationMinimum={setSaturationMinimum}
              setIntensityUpperLimit={setIntensityUpperLimit}
              setIntensityWeakThreshold={setIntensityWeakThreshold}
              setIntensityStrongThreshold={setIntensityStrongThreshold}
              setIntensityLowerLimit={setIntensityLowerLimit}
            />
          </div>

          <div style={{ marginTop: '1rem', fontSize: '0.875rem', color: '#333' }}>
            <p><strong>Name:</strong> {selectedItem.name || 'N/A'}</p>
            {itemId && <p><strong>ID:</strong> {itemId}</p>}
            {histogramData?.tissue_analysis && (
              <p style={{ marginTop: '0.5rem' }}>
                <strong>Tissue:</strong> {histogramData.tissue_analysis.tissue_percentage.toFixed(1)}% | 
                <strong> Background:</strong> {(100 - histogramData.tissue_analysis.tissue_percentage).toFixed(1)}%
              </p>
            )}
            <ViewportInfo
              currentViewport={currentViewport}
              viewportRegion={lastCapturedRegion}
              imageDimensions={imageDimensions}
            />
          </div>
        </div>
        
        {/* Region Display Panel - Right side */}
        <CapturedRegionPanel
          capturedRegions={capturedRegions}
          setCapturedRegions={setCapturedRegions}
          hueValue={hueValue}
          hueWidth={hueWidth}
          saturationMinimum={saturationMinimum}
          intensityUpperLimit={intensityUpperLimit}
          intensityWeakThreshold={intensityWeakThreshold}
          intensityStrongThreshold={intensityStrongThreshold}
          intensityLowerLimit={intensityLowerLimit}
          showWeak={showWeak}
          showPlain={showPlain}
          showStrong={showStrong}
          labelColorScheme={labelColorScheme}
        />
        
        {/* Right sidebar (Histogram + PPC) */}
        {isRightSidebarCollapsed ? (
          <div style={{ width: '36px', flex: '0 0 36px' }}>
            <button
              onClick={() => setIsRightSidebarCollapsed(false)}
              title="Show analysis sidebar"
              aria-label="Show analysis sidebar"
              style={{
                width: '36px',
                height: '36px',
                borderRadius: '8px',
                border: '1px solid #ddd',
                backgroundColor: '#fff',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#333',
              }}
            >
              ◀
            </button>
          </div>
        ) : (
          <div style={{ 
            flex: '1 1 40%',
            display: 'flex',
            flexDirection: 'column',
            gap: '1rem',
            minWidth: '300px'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ fontWeight: 700, color: '#333' }}>Analysis</div>
              <button
                onClick={() => setIsRightSidebarCollapsed(true)}
                title="Hide analysis sidebar"
                aria-label="Hide analysis sidebar"
                style={{
                  width: '36px',
                  height: '36px',
                  borderRadius: '8px',
                  border: '1px solid #ddd',
                  backgroundColor: '#fff',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#333',
                }}
              >
                ▶
              </button>
            </div>

            {/* Histogram */}
            <HistogramSection
              histogramData={histogramData}
              histogramLoading={histogramLoading}
              minWidth="300px"
            />

            {/* PPC Section (collapsible) */}
            <div style={{ 
              border: '1px solid #ddd', 
              borderRadius: '8px', 
              padding: '1rem',
              backgroundColor: '#fff'
            }}>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  cursor: 'pointer',
                  userSelect: 'none',
                  marginBottom: isPpcPanelCollapsed ? 0 : '1rem',
                }}
                onClick={() => setIsPpcPanelCollapsed(!isPpcPanelCollapsed)}
                title={isPpcPanelCollapsed ? 'Click to expand PPC controls' : 'Click to collapse PPC controls'}
              >
                <h3 style={{ margin: 0, color: '#333' }}>Positive Pixel Count (HSI Method)</h3>
                <span
                  style={{
                    fontSize: '0.875rem',
                    color: '#666',
                    transition: 'transform 0.2s',
                    transform: isPpcPanelCollapsed ? 'rotate(-90deg)' : 'rotate(0deg)',
                    display: 'inline-block',
                  }}
                >
                  ▼
                </span>
              </div>

              {!isPpcPanelCollapsed && (
                <div>
                  {/* HSI Parameters */}
                  <PpcParameters
                    itemId={itemId}
                    hueValue={hueValue}
                    hueWidth={hueWidth}
                    saturationMinimum={saturationMinimum}
                    intensityUpperLimit={intensityUpperLimit}
                    intensityWeakThreshold={intensityWeakThreshold}
                    intensityStrongThreshold={intensityStrongThreshold}
                    intensityLowerLimit={intensityLowerLimit}
                    setHueValue={setHueValue}
                    setHueWidth={setHueWidth}
                    setSaturationMinimum={setSaturationMinimum}
                    setIntensityUpperLimit={setIntensityUpperLimit}
                    setIntensityWeakThreshold={setIntensityWeakThreshold}
                    setIntensityStrongThreshold={setIntensityStrongThreshold}
                    setIntensityLowerLimit={setIntensityLowerLimit}
                    roiHueReport={roiHueReport}
                    setRoiHueReport={setRoiHueReport}
                    onAfterRoiSample={() => setShowSampledRois(true)}
                    roiSamplingMode={roiSamplingMode}
                    setRoiSamplingMode={setRoiSamplingMode}
                  />

                  <AutoComputeControls
                    autoCompute={autoCompute}
                    setAutoCompute={setAutoCompute}
                    intensityMap={intensityMap}
                  />
                  <button
                    onClick={() => computePpc(itemId)}
                    disabled={ppcLoading || !selectedItem}
                    style={{
                      padding: '0.5rem 1rem',
                      fontSize: '0.875rem',
                      backgroundColor: autoCompute ? '#2196F3' : '#4CAF50',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: ppcLoading || !selectedItem ? 'not-allowed' : 'pointer',
                      opacity: ppcLoading || !selectedItem ? 0.6 : 1,
                      width: '100%',
                      marginBottom: '1rem'
                    }}
                  >
                    {ppcLoading || intensityMapLoading ? 'Computing...' : autoCompute ? 'Force Compute Now' : 'Compute PPC'}
                  </button>
                  <PpcResults
                    ppcData={ppcData}
                    ppcLoading={ppcLoading}
                    intensityMapLoading={intensityMapLoading}
                    intensityMap={intensityMap}
                  />
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
