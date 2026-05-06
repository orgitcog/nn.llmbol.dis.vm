# Document Upload for Knowledge Feature Implementation

## Overview

This implementation adds document upload for knowledge functionality to the nn.llmbol.dis.vm project.
Users can now upload text-based reference documents — source files, configuration files, markdown notes,
style guides, etc. — directly in the chat input. The content of these documents is automatically injected
as context into the next AI message, so the model can reference them when generating code or answering
questions.

## Modified Files

### 1. `app/components/chat/FilePreview.tsx`

- Extended to display non-image document files with appropriate file-type icons (`i-ph:file-text`,
  `i-ph:file-code`, `i-ph:file-js`, `i-ph:file-css`, `i-ph:file-html`, etc.)
- Added `documentContentList` prop (parallel to `imageDataList`) so document slots are correctly
  identified and shown in the preview strip
- Preserved existing image thumbnail behaviour untouched

### 2. `app/components/chat/BaseChat.tsx`

- Added `documentContentList` and `setDocumentContentList` to `BaseChatProps`
- Rewrote `handleFileUpload` to accept both images and a wide set of text-based document extensions
  (`.txt`, `.md`, `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.env`, `.csv`, plus common source-code
  extensions) using `multiple` file selection
- For image files the existing base64 + `imageDataList` path is used; for document files the text
  content is read and stored in `documentContentList`
- Updated the paste handler to keep `documentContentList` in sync when images are pasted
- Passes the new props through to `ChatBox`

### 3. `app/components/chat/ChatBox.tsx`

- Added `documentContentList` and `setDocumentContentList` to the component's props interface
- Extended the drag-and-drop handler to recognise document files and read them as text, mirroring
  the new upload-button behaviour
- Updated `FilePreview` usage to pass `documentContentList` and to remove the matching entry from all
  three parallel arrays (`uploadedFiles`, `imageDataList`, `documentContentList`) on removal

### 4. `app/components/chat/Chat.client.tsx`

- Added `documentContentList` / `setDocumentContentList` state
- Added `buildDocumentContext` helper: produces a `<knowledge_documents>` XML block containing each
  document's content in a fenced code block labelled with the file extension
- Modified `sendMessage` to prepend the document context block to `finalMessageContent` before
  building any message text, covering all send paths (first message, template flow, modified-files
  path, and regular append)
- `imageDataList.filter(Boolean)` ensures only image slots (non-empty entries) are forwarded as file
  parts to the AI SDK
- Clears `documentContentList` alongside `uploadedFiles` and `imageDataList` after every send

## Key Features

1. **Multi-file selection**: the upload button now opens a picker that accepts images *and* a broad
   set of document extensions in a single dialog
2. **Drag-and-drop documents**: dropping text/code files onto the chat textarea now works the same
   as using the upload button
3. **Visual document previews**: uploaded documents appear in the preview strip with recognisable
   file-type icons and the filename, consistent with the existing image thumbnails
4. **Automatic context injection**: document contents are wrapped in `<knowledge_documents>` /
   `<document>` tags and prepended to the user message so every LLM provider receives them as
   plain text — no provider-specific attachment support required
5. **Per-message scope**: documents are cleared after each send, preventing unintended repetition
   in subsequent messages

---



## Overview

This implementation adds three new LLM provider integrations to the nn.llmbol.dis.vm system, completing the "Additional Provider Integrations" milestone listed in the project roadmap. Each provider follows the established `BaseProvider` pattern and is automatically discovered by `LLMManager` at startup.

## New Files

### 1. `app/lib/modules/llm/providers/azure-openai.ts`

- Integrates Azure OpenAI Service using `@ai-sdk/azure`
- Supports GPT-4o, GPT-4o Mini, GPT-4 Turbo, and GPT-3.5 Turbo deployments
- Configurable via `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_RESOURCE_NAME`, and `AZURE_OPENAI_API_VERSION`
- Defaults to API version `2024-10-01-preview`

### 2. `app/lib/modules/llm/providers/vertex-ai.ts`

- Integrates Google Cloud Vertex AI using `@ai-sdk/google-vertex`
- Exposes Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 2.0 Flash, and Gemini 2.0 Pro models
- Configurable via `VERTEX_AI_PROJECT` and `VERTEX_AI_LOCATION` (defaults to `us-central1`)
- Uses Application Default Credentials for Google Cloud authentication

### 3. `app/lib/modules/llm/providers/granite.ts`

- Integrates IBM Granite foundation models via IBM watsonx.ai
- Static models include Granite 3 8B Instruct, Granite 3 2B Instruct, Granite 20B Multilingual, Granite 34B Code Instruct, and Granite 8B Code Instruct
- Dynamic model discovery queries the watsonx.ai foundation model catalog
- Configurable via `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, and `WATSONX_BASE_URL`
- Accesses the OpenAI-compatible chat completions endpoint on watsonx.ai

## Modified Files

### 1. `app/lib/modules/llm/registry.ts`

- Registered `AzureOpenAIProvider`, `VertexAIProvider`, and `GraniteProvider`

### 2. `package.json`

- Added `@ai-sdk/azure@1.0.7` dependency for Azure OpenAI support
- Added `@ai-sdk/google-vertex@2.0.7` dependency for Vertex AI support

### 3. `.env.example`

- Added configuration examples for Azure OpenAI (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_RESOURCE_NAME`, `AZURE_OPENAI_API_VERSION`)
- Added configuration examples for Vertex AI (`VERTEX_AI_PROJECT`, `VERTEX_AI_LOCATION`)
- Added configuration examples for IBM watsonx.ai (`WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, `WATSONX_BASE_URL`)

### 4. `app/lib/modules/__tests__/llm.spec.ts`

- Extended with test suites for `AzureOpenAIProvider`, `VertexAIProvider`, and `GraniteProvider`
- Tests cover provider metadata, static model listings, API key link validity, config keys, and error handling

## Key Features

1. **Azure OpenAI**: Full enterprise-grade Azure OpenAI integration with configurable API version and deployment names as model identifiers
2. **Vertex AI**: Google Cloud Vertex AI integration with support for all current Gemini 2.x models and Application Default Credentials
3. **IBM Granite**: Open-weight Granite foundation models including specialized code models through the IBM watsonx.ai platform
4. **Automatic Registration**: All three providers are auto-registered by `LLMManager` via the provider registry pattern
5. **Dynamic Discovery**: `GraniteProvider` includes dynamic model listing from the watsonx.ai foundation model catalog

---

# File and Folder Locking Feature Implementation

## Overview

This implementation adds persistent file and folder locking functionality to the BoltDIY project. When a file or folder is locked, it cannot be modified by either the user or the AI until it is unlocked. All locks are scoped to the current chat/project to prevent locks from one project affecting files with matching names in other projects.

## New Files

### 1. `app/components/chat/LockAlert.tsx`

- A dedicated alert component for displaying lock-related error messages
- Features a distinctive amber/yellow color scheme and lock icon
- Provides clear instructions to the user about locked files

### 2. `app/lib/persistence/lockedFiles.ts`

- Core functionality for persisting file and folder locks in localStorage
- Provides functions for adding, removing, and retrieving locked files and folders
- Defines the lock modes: "full" (no modifications) and "scoped" (only additions allowed)
- Implements chat ID scoping to isolate locks to specific projects

### 3. `app/utils/fileLocks.ts`

- Utility functions for checking if a file or folder is locked
- Helps avoid circular dependencies between components and stores
- Provides a consistent interface for lock checking across the application
- Extracts chat ID from URL for project-specific lock scoping

## Modified Files

### 1. `app/components/chat/ChatAlert.tsx`

- Updated to use the new LockAlert component for locked file errors
- Maintains backward compatibility with other error types

### 2. `app/components/editor/codemirror/CodeMirrorEditor.tsx`

- Added checks to prevent editing of locked files
- Updated to use the new fileLocks utility
- Displays appropriate tooltips when a user attempts to edit a locked file

### 3. `app/components/workbench/EditorPanel.tsx`

- Added safety checks for unsavedFiles to prevent errors
- Improved handling of locked files in the editor panel

### 4. `app/components/workbench/FileTree.tsx`

- Added visual indicators for locked files and folders in the file tree
- Improved handling of locked files and folders in the file tree
- Added context menu options for locking and unlocking folders

### 5. `app/lib/stores/editor.ts`

- Added checks to prevent updating locked files
- Improved error handling for locked files

### 6. `app/lib/stores/files.ts`

- Added core functionality for locking and unlocking files and folders
- Implemented persistence of locked files and folders across page refreshes
- Added methods for checking if a file or folder is locked
- Added chat ID scoping to prevent locks from affecting other projects

### 7. `app/lib/stores/workbench.ts`

- Added methods for locking and unlocking files and folders
- Improved error handling for locked files and folders
- Fixed issues with alert initialization
- Added support for chat ID scoping of locks

### 8. `app/types/actions.ts`

- Added `isLockedFile` property to the ActionAlert interface
- Improved type definitions for locked file alerts

## Key Features

1. **Persistent File and Folder Locking**: Locks are stored in localStorage and persist across page refreshes
2. **Visual Indicators**: Locked files and folders are clearly marked in the UI with lock icons
3. **Improved Error Messages**: Clear, visually distinct error messages when attempting to modify locked items
4. **Lock Modes**: Support for both full locks (no modifications) and scoped locks (only additions allowed)
5. **Prevention of AI Modifications**: The AI is prevented from modifying locked files and folders
6. **Project-Specific Locks**: Locks are scoped to the current chat/project to prevent conflicts
7. **Recursive Folder Locking**: Locking a folder automatically locks all files and subfolders within it

## UI Improvements

1. **Enhanced Alert Design**: Modern, visually appealing alert design with better spacing and typography
2. **Contextual Icons**: Different icons and colors for different types of alerts
3. **Improved Error Details**: Better formatting of error details with monospace font and left border
4. **Responsive Buttons**: Better positioned and styled buttons with appropriate hover effects
