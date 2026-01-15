interface ViewportInfoProps {
  imageDimensions: { width: number, height: number } | null
  currentViewport: { x: number, y: number, width: number, height: number, zoom: number } | null
  viewportRegion: { x: number, y: number, width: number, height: number } | null
}

export function ViewportInfo({ imageDimensions, currentViewport, viewportRegion }: ViewportInfoProps) {
  if (!currentViewport && !viewportRegion && !imageDimensions) return null

  return (
    <div style={{ marginTop: '0.5rem', padding: '0.75rem', backgroundColor: '#f0f8ff', borderRadius: '4px', border: '1px solid #b3d9ff' }}>
      <p style={{ margin: '0 0 0.5rem 0', fontWeight: 'bold', color: '#333', fontSize: '0.875rem' }}>Viewport Region:</p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.75rem', color: '#333' }}>
        {imageDimensions && (
          <div>
            <strong style={{ color: '#222' }}>Image:</strong>{' '}
            <span style={{ color: '#555' }}>
              {imageDimensions.width.toLocaleString()} × {imageDimensions.height.toLocaleString()} px
            </span>
          </div>
        )}
        {currentViewport && (
          <>
            <div>
              <strong style={{ color: '#222' }}>Viewport (normalized):</strong>{' '}
              <span style={{ color: '#555', fontFamily: 'monospace' }}>
                X: {currentViewport.x.toFixed(4)}, Y: {currentViewport.y.toFixed(4)}, 
                W: {currentViewport.width.toFixed(4)}, H: {currentViewport.height.toFixed(4)}
              </span>
            </div>
            {imageDimensions && (
              <div>
                <strong style={{ color: '#222' }}>Viewport (pixels):</strong>{' '}
                <span style={{ color: '#555', fontFamily: 'monospace' }}>
                  X: {Math.round(currentViewport.x * imageDimensions.width).toLocaleString()}, 
                  Y: {Math.round(currentViewport.y * imageDimensions.height).toLocaleString()}, 
                  W: {Math.round(currentViewport.width * imageDimensions.width).toLocaleString()}, 
                  H: {Math.round(currentViewport.height * imageDimensions.height).toLocaleString()}
                </span>
              </div>
            )}
            <div>
              <strong style={{ color: '#222' }}>Zoom:</strong>{' '}
              <span style={{ color: '#555', fontWeight: 'bold' }}>{currentViewport.zoom.toFixed(2)}×</span>
            </div>
            {imageDimensions && currentViewport && (
              <div>
                <strong style={{ color: '#222' }}>Region:</strong>{' '}
                <span style={{ color: '#555', fontWeight: 'bold' }}>
                  {((currentViewport.width * currentViewport.height) * 100).toFixed(2)}%
                </span>
              </div>
            )}
          </>
        )}
        {viewportRegion && !currentViewport && (
          <div>
            <strong style={{ color: '#222' }}>Captured Region:</strong>{' '}
            <span style={{ color: '#555', fontFamily: 'monospace' }}>
              x: {viewportRegion.x.toFixed(4)}, y: {viewportRegion.y.toFixed(4)}, 
              w: {viewportRegion.width.toFixed(4)}, h: {viewportRegion.height.toFixed(4)}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
