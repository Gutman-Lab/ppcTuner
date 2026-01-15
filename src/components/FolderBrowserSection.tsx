import { FolderBrowser, ThumbnailGrid } from 'bdsa-react-components'
import type { Resource, Item } from 'bdsa-react-components'

interface FolderBrowserSectionProps {
  configLoaded: boolean
  apiBaseUrl: string
  dsaToken: string | null
  selectedFolderId: string
  startFolder?: string
  startFolderType?: 'collection' | 'folder'
  folderBrowserApiHeaders?: HeadersInit
  folderBrowserFetchFn?: (url: string, options?: RequestInit) => Promise<Response>
  thumbnailGridApiHeaders?: HeadersInit
  thumbnailGridFetchFn?: (url: string, options?: RequestInit) => Promise<Response>
  onResourceSelect: (resource: Resource) => void
  onThumbnailClick: (item: Item) => void
}

export function FolderBrowserSection({
  configLoaded,
  apiBaseUrl,
  dsaToken,
  selectedFolderId,
  startFolder,
  startFolderType,
  folderBrowserApiHeaders,
  folderBrowserFetchFn,
  thumbnailGridApiHeaders,
  thumbnailGridFetchFn,
  onResourceSelect,
  onThumbnailClick,
}: FolderBrowserSectionProps) {
  return (
    <>
      <div className="browser-section">
        <h2 style={{ margin: 0, marginBottom: '0.5rem' }}>Select Folder</h2>
        {configLoaded && (
          <FolderBrowser
            apiBaseUrl={apiBaseUrl}
            showCollections={true}
            onResourceSelect={onResourceSelect}
            foldersPerPage={50}
            {...(startFolder 
              ? { 
                  rootId: startFolder,
                  rootType: startFolderType as 'collection' | 'folder'
                } 
              : {})}
            apiHeaders={folderBrowserApiHeaders}
            fetchFn={folderBrowserFetchFn}
          />
        )}
      </div>

      {selectedFolderId ? (
        <div className="thumbnail-section">
          <h2>Thumbnails</h2>
          {!dsaToken && (
            <div style={{ padding: '1rem', backgroundColor: '#fff3cd', border: '1px solid #ffc107', borderRadius: '4px', marginBottom: '1rem' }}>
              <strong>⚠ Authentication Warning:</strong> No DSA token available. Thumbnail requests may fail with 401 errors. Check backend logs for token initialization.
            </div>
          )}
          <ThumbnailGrid
            key={`${selectedFolderId}-${dsaToken ? 'auth' : 'noauth'}`}
            apiBaseUrl={apiBaseUrl}
            folderId={selectedFolderId}
            thumbnailSize="l"
            itemsPerPage={12}
            apiHeaders={thumbnailGridApiHeaders}
            tokenQueryParam={!!dsaToken}
            fetchFn={thumbnailGridFetchFn}
            onThumbnailClick={onThumbnailClick}
          />
        </div>
      ) : (
        <div className="thumbnail-section" style={{ opacity: 0.5 }}>
          <h2>Thumbnails</h2>
          <p>Select a folder to view thumbnails</p>
        </div>
      )}
    </>
  )
}
