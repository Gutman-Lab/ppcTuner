# bdsa-react-components Library Improvement Suggestions

This document outlines features and improvements that would be useful to add to the `bdsa-react-components` library based on real-world usage in the PPC Tuner V2 application.

## 1. Initial Folder/Collection Navigation

### Current State
- `FolderBrowser` component has `rootId` and `rootType` props (confirmed in Storybook)
- These props are not well documented in the main documentation
- `rootType` accepts `'collection'` or `'folder'`
- `rootId` is the ID of the collection or folder to start at

### Proposed Enhancement
Add props to allow `FolderBrowser` to start at a specific folder or collection:

**Note:** The component already has `rootId` and `rootType` props (confirmed in Storybook controls). These should be better documented.

```tsx
<FolderBrowser
  apiBaseUrl="http://bdsa.pathology.emory.edu:8080/api/v1"
  showCollections={true}
  // Existing props (needs better documentation):
  rootId="695d6a148c871f3a02969b00"           // ID of collection or folder to start at
  rootType="collection"                        // 'collection' or 'folder'
  onResourceSelect={(resource) => console.log(resource)}
/>
```

**Alternative approach:** Unified prop that accepts both types:
```tsx
<FolderBrowser
  apiBaseUrl={...}
  initialResource={{
    type: 'collection' | 'folder',
    id: string
  }}
  // Automatically navigates to and displays the specified resource on mount
/>
```

**Benefits:**
- Better user experience - users land directly at the relevant folder/collection
- Reduces navigation steps
- Useful for applications that have a "default" or "start" folder/collection
- Supports deep linking scenarios

**Implementation Notes:**
- Should navigate to the specified resource on component mount
- Should update breadcrumb/navigation state to reflect the starting point
- Should work with both collections and folders
- Should handle cases where the resource doesn't exist or user doesn't have access

---

## 2. API Key → Token Authentication Pattern

### Current State
- `DsaAuthManager` component exists but documentation is minimal ("Callback when authentication status changes")
- No built-in support for API key → token exchange
- Applications must implement this pattern manually in their backend

### Proposed Enhancement
Add support for automatic API key → token authentication:

**Option A: Backend Endpoint Pattern**
```tsx
<DsaAuthManager
  apiBaseUrl="http://bdsa.pathology.emory.edu:8080/api/v1"
  tokenEndpoint="/api/config"  // Backend endpoint that returns { dsaToken: string }
  onTokenChange={(token) => {
    // Token obtained, ready to use
  }}
>
  {/* Child components automatically receive token via context */}
  <FolderBrowser apiBaseUrl={...} />
  <ThumbnailGrid apiBaseUrl={...} />
</DsaAuthManager>
```

**Option B: Direct API Key Pattern**
```tsx
<DsaAuthManager
  apiBaseUrl="http://bdsa.pathology.emory.edu:8080/api/v1"
  apiKey="your-api-key-here"
  onTokenChange={(token) => {
    // Token obtained from API key
  }}
>
  {/* Child components */}
</DsaAuthManager>
```

**Benefits:**
- Simplifies authentication setup for applications
- Keeps API keys secure (Option A keeps key on backend)
- Automatic token refresh/re-authentication
- Context-based token distribution to child components

**Implementation Notes:**
- Use React Context to provide token to all child components
- Handle token expiration and refresh automatically
- Support both patterns (backend endpoint and direct API key)

---

## 3. Enhanced DsaAuthManager Documentation

### Current State
- Minimal documentation: "Callback when authentication status changes"
- No examples or prop descriptions

### Proposed Enhancement
Add comprehensive documentation including:
- All available props with types and descriptions
- Usage examples for different authentication patterns
- Integration with other components (FolderBrowser, ThumbnailGrid, etc.)
- Error handling patterns
- Token refresh strategies

---

## 4. Token Context Provider

### Current State
- Each component requires `apiHeaders` or `fetchFn` prop individually
- No centralized token management

### Proposed Enhancement
Create a `DsaAuthProvider` context that:
- Manages authentication state centrally
- Automatically injects token into all child component requests
- Provides hooks for accessing auth state: `useDsaAuth()`, `useDsaToken()`

**Example:**
```tsx
import { DsaAuthProvider, useDsaToken } from 'bdsa-react-components'

function App() {
  return (
    <DsaAuthProvider
      apiBaseUrl="http://bdsa.pathology.emory.edu:8080/api/v1"
      tokenEndpoint="/api/config"
    >
      <MyComponent />
    </DsaAuthProvider>
  )
}

function MyComponent() {
  const { token, isAuthenticated, user } = useDsaToken()
  // Components automatically use token - no need to pass apiHeaders
  return <FolderBrowser apiBaseUrl={...} />
}
```

**Benefits:**
- Reduces boilerplate (no need to pass `apiHeaders` to every component)
- Centralized authentication state
- Easier to implement authentication UI (login/logout buttons, user info)

---

## 5. Error Handling Improvements

### Current State
- Components may fail silently or show generic errors
- No standardized error handling for authentication failures

### Proposed Enhancement
Add standardized error handling:
- `onAuthError` callback for authentication failures
- `onApiError` callback for API errors
- Error boundary components for graceful error display
- Retry logic for transient failures

**Example:**
```tsx
<DsaAuthProvider
  apiBaseUrl={...}
  onAuthError={(error) => {
    console.error('Authentication failed:', error)
    // Redirect to login, show error message, etc.
  }}
  onApiError={(error, retry) => {
    if (error.status === 401) {
      // Token expired, refresh
      retry()
    }
  }}
>
```

---

## 6. Loading States and Skeletons

### Current State
- Components may not provide clear loading indicators
- No standardized loading UI

### Proposed Enhancement
- Built-in loading skeletons for FolderBrowser, ThumbnailGrid
- Loading state callbacks: `onLoadingChange={(isLoading) => ...}`
- Configurable loading UI (skeletons, spinners, custom components)

---

## 7. TypeScript Type Improvements

### Current State
- Types exist but could be more comprehensive

### Proposed Enhancement
- Export all internal types for advanced use cases
- Add JSDoc comments to all exported types
- Provide type guards for runtime type checking
- Add discriminated unions for better type narrowing

---

## 8. Configuration Presets

### Current State
- Each application must configure DSA base URL, authentication, etc. individually

### Proposed Enhancement
Create configuration presets for common DSA instances:
```tsx
import { DsaConfig } from 'bdsa-react-components'

const emoryDsa = DsaConfig.preset('emory', {
  apiBaseUrl: 'http://bdsa.pathology.emory.edu:8080/api/v1',
  // Other common settings
})

<DsaAuthProvider config={emoryDsa} ... />
```

---

## 9. Development Tools

### Proposed Enhancement
- React DevTools integration for debugging auth state
- Console logging utilities (optional, for development)
- Mock data generators for testing
- Storybook stories for all components

---

## 10. Performance Optimizations

### Proposed Enhancement
- Request deduplication (multiple components requesting same resource)
- Built-in caching strategies
- Virtual scrolling for large folder/item lists
- Image lazy loading for thumbnails

---

## 11. Testing Utilities

### Proposed Enhancement
- Mock DSA server for testing
- Test utilities for common scenarios
- Example test suites
- Integration test helpers

---

## Priority Recommendations

### High Priority
1. **Initial Folder/Collection Navigation** - Common use case, improves UX
2. **API Key → Token Authentication Pattern** - Most requested feature
3. **Enhanced DsaAuthManager Documentation** - Critical for adoption
4. **Token Context Provider** - Reduces boilerplate significantly

### Medium Priority
4. **Error Handling Improvements** - Better developer experience
5. **Loading States and Skeletons** - Better user experience
6. **TypeScript Type Improvements** - Better developer experience

### Low Priority
7. **Configuration Presets** - Nice to have
8. **Development Tools** - Nice to have
9. **Performance Optimizations** - Optimization, not critical
10. **Testing Utilities** - Helpful but not blocking

---

## Implementation Notes

### Backward Compatibility
- All new features should be opt-in
- Existing `apiHeaders` and `fetchFn` props should continue to work
- Context provider should be optional (components work standalone)

### Migration Path
- Provide migration guide for existing applications
- Deprecation warnings for old patterns (with clear migration path)
- Examples showing both old and new patterns

### Testing Strategy
- Unit tests for all new features
- Integration tests with real DSA server
- E2E tests for common workflows
- Performance benchmarks

---

## Related Issues/PRs

When implementing these features, consider:
- Security: API keys should never be exposed to frontend (use backend endpoint pattern)
- Performance: Token caching and refresh strategies
- Accessibility: All UI components should be accessible
- Internationalization: Error messages and UI text should be i18n-ready

---

## Example: Complete Implementation Pattern

Here's how a complete implementation might look:

```tsx
import {
  DsaAuthProvider,
  FolderBrowser,
  ThumbnailGrid,
  useDsaAuth,
} from 'bdsa-react-components'

function App() {
  return (
    <DsaAuthProvider
      apiBaseUrl="http://bdsa.pathology.emory.edu:8080/api/v1"
      tokenEndpoint="/api/config"
      onAuthError={(error) => {
        console.error('Auth error:', error)
        // Show error UI
      }}
    >
      <AppContent />
    </DsaAuthProvider>
  )
}

function AppContent() {
  const { isAuthenticated, user, token } = useDsaAuth()
  
  if (!isAuthenticated) {
    return <div>Loading authentication...</div>
  }
  
  return (
    <div>
      <p>Logged in as: {user?.login}</p>
      <FolderBrowser
        apiBaseUrl="http://bdsa.pathology.emory.edu:8080/api/v1"
        // Token automatically injected via context
        onResourceSelect={(resource) => {
          console.log('Selected:', resource)
        }}
      />
    </div>
  )
}
```

---

## 12. SlideViewer Viewport Coordinates Callback

### Current State
- `SlideViewer` component wraps OpenSeadragon but does not expose viewport coordinates
- Applications must access OpenSeadragon's global registry or DOM to get viewport bounds
- This is fragile and requires knowledge of OpenSeadragon internals

### Proposed Enhancement
Add `onViewportChange` callback prop to `SlideViewer` that fires whenever the viewport changes (pan, zoom, resize):

```tsx
<SlideViewer
  imageInfo={{ dziUrl: '...' }}
  height="600px"
  onViewportChange={(bounds) => {
    // bounds: { x: number, y: number, width: number, height: number, zoom: number }
    // Coordinates are normalized (0-1) relative to full image
    console.log('Viewport:', bounds)
    setViewportState(bounds)
  }}
/>
```

**Callback signature:**
```tsx
type ViewportBounds = {
  x: number        // Left edge (0-1, normalized)
  y: number        // Top edge (0-1, normalized)
  width: number    // Width (0-1, normalized)
  height: number   // Height (0-1, normalized)
  zoom: number     // Current zoom level (e.g., 1.0 = 100%)
}

onViewportChange?: (bounds: ViewportBounds) => void
```

**Implementation Notes:**
- Fire callback on OpenSeadragon events: `animation`, `pan`, `zoom`, `resize`
- Debounce rapid updates (e.g., during animations) to avoid excessive callbacks
- Provide normalized coordinates (0-1) for consistency across different image sizes
- Include zoom level for convenience
- Fire callback immediately after viewer initialization with initial viewport state

**Benefits:**
- Clean API - no need to access OpenSeadragon internals
- Consistent coordinate system (normalized 0-1)
- Enables features like:
  - Viewport coordinate displays
  - Region capture/cropping
  - Viewport-based annotations
  - Viewport history/undo-redo
  - Viewport sharing/bookmarking

**Example Use Cases:**
- Display current viewport coordinates in UI
- Capture a region of the image based on current viewport
- Save/restore viewport state
- Sync viewport across multiple viewers
- Generate viewport-based annotations

---

**Generated:** 2025-01-09  
**Based on:** PPC Tuner V2 implementation experience  
**Status:** Suggestions for library maintainers
