import { useState, useEffect } from 'react'
import './App.css'
import 'bdsa-react-components/styles.css'
import type { Resource, Item } from 'bdsa-react-components'
import { FolderBrowserSection } from './components/FolderBrowserSection'
import { ImageViewerSection } from './components/ImageViewerSection'
import type { SavedParameterSet, PpcResult, CapturedRegion } from './types'
import { useLocalStorage } from './hooks/useLocalStorage'
import { useDsaAuth } from './hooks/useDsaAuth'
import { usePpc } from './hooks/usePpc'
import { useAutoDetectHue } from './hooks/useAutoDetectHue'
import { useImageSelection } from './hooks/useImageSelection'

function App() {
  // Load saved state from localStorage
  const [selectedFolderId, setSelectedFolderId] = useLocalStorage<string>('ppcTuner_selectedFolderId', '')
  const [selectedItem, setSelectedItem] = useLocalStorage<Item | null>('ppcTuner_selectedItem', null)
  const [histogramData, setHistogramData] = useState<any>(null)
  const [histogramLoading, setHistogramLoading] = useState(false)
  const [showTissueMask, setShowTissueMask] = useState(false)
  
  // OpenSeadragon viewer state
  const [capturedRegions, setCapturedRegions] = useState<CapturedRegion[]>([])
  const [imageDimensions, setImageDimensions] = useState<{ width: number, height: number } | null>(null)
  const [currentViewport, setCurrentViewport] = useState<{ x: number, y: number, width: number, height: number, zoom: number } | null>(null)
  // HSI parameters (declared early to avoid initialization order issues)
  const [hueValue, setHueValue] = useState<number>(0.1)
  const [hueWidth, setHueWidth] = useState<number>(0.1)
  const [saturationMinimum, setSaturationMinimum] = useState<number>(0.1)
  const [intensityUpperLimit, setIntensityUpperLimit] = useState<number>(0.9)
  const [intensityWeakThreshold, setIntensityWeakThreshold] = useState<number>(0.6)
  const [intensityStrongThreshold, setIntensityStrongThreshold] = useState<number>(0.3)
  const [intensityLowerLimit, setIntensityLowerLimit] = useState<number>(0.05)
  
  const [showPpcLabel, setShowPpcLabel] = useState<boolean>(false)
  const [ppcLabelOpacity, setPpcLabelOpacity] = useState<number>(0.5)
  const [showWeak, setShowWeak] = useState<boolean>(true)
  const [showPlain, setShowPlain] = useState<boolean>(true)
  const [showStrong, setShowStrong] = useState<boolean>(true)
  const [labelColorScheme, setLabelColorScheme] = useState<string>('blue-green-red')
  
  // Real-time adjustment mode
  const [autoCompute, setAutoCompute] = useState<boolean>(false)
  const [savedParameterSets, setSavedParameterSets] = useLocalStorage<SavedParameterSet[]>('ppcTuner_savedParameterSets', [])
  const [nextSetId, setNextSetId] = useLocalStorage<number>('ppcTuner_nextSetId', 1)

  // Use PPC hook for computation and intensity map management
  const {
    ppcData,
    setPpcData,
    ppcLoading,
    intensityMap,
    setIntensityMap,
    intensityMapLoading,
    computePpc,
  } = usePpc({
    hueValue,
    hueWidth,
    saturationMinimum,
    intensityUpperLimit,
    intensityWeakThreshold,
    intensityStrongThreshold,
    intensityLowerLimit,
    autoCompute,
    selectedItem,
  })

  // Note: selectedFolderId and selectedItem are automatically persisted by useLocalStorage hook
  // Image dimensions fetching is handled in useImageSelection hook
  
  // Use DSA authentication hook
  const {
    apiBaseUrl,
    dsaToken,
    configLoaded,
    startFolder,
    startFolderType,
    thumbnailGridFetchFn,
    folderBrowserFetchFn,
    thumbnailGridApiHeaders,
    folderBrowserApiHeaders,
  } = useDsaAuth()
  
  // Use auto-detect hue hook
  const { autoDetectHue } = useAutoDetectHue({
    setHueValue,
    setHueWidth,
    setSaturationMinimum,
    setIntensityUpperLimit,
    setIntensityWeakThreshold,
    setIntensityStrongThreshold,
    setIntensityLowerLimit,
  })

  // Use image selection hook
  const { handleThumbnailClick } = useImageSelection({
    selectedItem,
    configLoaded,
    dsaToken,
    setSelectedItem,
    setHistogramData,
    setHistogramLoading,
    setPpcData,
    setIntensityMap,
    setShowPpcLabel,
    setCapturedRegions,
    setSavedParameterSets,
    setImageDimensions,
    setCurrentViewport,
    autoDetectHue,
  })

  // Note: we intentionally auto-detect hue once per slide selection (not per-region),
  // since ROI sampling showed it to be stable and re-detecting can be distracting.

  const handleResourceSelect = (resource: Resource) => {
    console.log('Resource selected:', resource)
    
    // Resource object uses _id (MongoDB/Girder convention), not id
    const resourceId = (resource as any)._id || (resource as any).id
    const resourceType = (resource as any)._modelType || resource.type
    
    console.log('Resource type:', resourceType)
    console.log('Resource ID:', resourceId, 'Type:', typeof resourceId)
    
    // Handle both folders and collections - collections can contain items too
    if ((resourceType === 'folder' || resource.type === 'folder') && typeof resourceId === 'string') {
      console.log('✓ Setting selected folder ID:', resourceId)
      setSelectedFolderId(resourceId)
    } else if ((resourceType === 'collection' || resource.type === 'collection') && typeof resourceId === 'string') {
      // Collections can also contain items, so allow selecting them
      console.log('✓ Setting selected collection ID:', resourceId)
      setSelectedFolderId(resourceId)
    } else {
      console.log('⚠ Resource is not a folder/collection or missing ID:', resource)
      setSelectedFolderId('')
    }
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>PPC Tuner</h1>
        <p>Load images from DSA and generate aBeta-stained slide thumbnails</p>
        {dsaToken && <p style={{ fontSize: '0.85rem', opacity: 0.8 }}>✓ Authenticated with DSA (token obtained from API key)</p>}
        {!dsaToken && configLoaded && <p style={{ fontSize: '0.85rem', color: '#e74c3c' }}>⚠ No authentication token - check DSAKEY in .env and backend logs</p>}
      </header>

      <main className="app-main">
        <FolderBrowserSection
          configLoaded={configLoaded}
          apiBaseUrl={apiBaseUrl}
          dsaToken={dsaToken}
          selectedFolderId={selectedFolderId}
          startFolder={startFolder}
          startFolderType={startFolderType as 'collection' | 'folder' | undefined}
          folderBrowserApiHeaders={folderBrowserApiHeaders}
          folderBrowserFetchFn={folderBrowserFetchFn}
          thumbnailGridApiHeaders={thumbnailGridApiHeaders}
          thumbnailGridFetchFn={thumbnailGridFetchFn}
          onResourceSelect={handleResourceSelect}
          onThumbnailClick={handleThumbnailClick}
        />

        {selectedItem && (() => {
          const itemId = (selectedItem as any)._id || (selectedItem as any).id
          if (!itemId) return null
          
          return (
            <ImageViewerSection
              selectedItem={selectedItem}
              itemId={itemId}
              apiBaseUrl={apiBaseUrl}
              dsaToken={dsaToken}
              histogramData={histogramData}
              histogramLoading={histogramLoading}
              imageDimensions={imageDimensions}
              currentViewport={currentViewport}
              setCurrentViewport={setCurrentViewport}
              capturedRegions={capturedRegions}
              setCapturedRegions={setCapturedRegions}
              ppcData={ppcData}
              ppcLoading={ppcLoading}
              intensityMap={intensityMap}
              intensityMapLoading={intensityMapLoading}
              computePpc={computePpc}
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
              showPpcLabel={showPpcLabel}
              setShowPpcLabel={setShowPpcLabel}
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
              showTissueMask={showTissueMask}
              setShowTissueMask={setShowTissueMask}
              autoCompute={autoCompute}
              setAutoCompute={setAutoCompute}
              savedParameterSets={savedParameterSets}
              setSavedParameterSets={setSavedParameterSets}
              nextSetId={nextSetId}
              setNextSetId={setNextSetId}
              autoDetectHue={autoDetectHue}
            />
          )
        })()}
      </main>
    </div>
  )
}

export default App
