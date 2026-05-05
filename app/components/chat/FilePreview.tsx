import React from 'react';

interface FilePreviewProps {
  files: File[];
  imageDataList: string[];
  documentContentList?: string[];
  onRemove: (index: number) => void;
}

const DOCUMENT_ICON_CLASSES: Record<string, string> = {
  json: 'i-ph:file-js',
  yaml: 'i-ph:file-code',
  yml: 'i-ph:file-code',
  toml: 'i-ph:file-code',
  md: 'i-ph:file-text',
  markdown: 'i-ph:file-text',
  txt: 'i-ph:file-text',
  ts: 'i-ph:file-code',
  tsx: 'i-ph:file-code',
  js: 'i-ph:file-js',
  jsx: 'i-ph:file-js',
  py: 'i-ph:file-code',
  css: 'i-ph:file-css',
  html: 'i-ph:file-html',
  xml: 'i-ph:file-code',
  sh: 'i-ph:terminal',
  env: 'i-ph:file-dotted',
  ini: 'i-ph:file-dotted',
  csv: 'i-ph:table',
};

function getDocumentIcon(filename: string): string {
  const ext = filename.split('.').pop()?.toLowerCase() ?? '';
  return DOCUMENT_ICON_CLASSES[ext] ?? 'i-ph:file';
}

const FilePreview: React.FC<FilePreviewProps> = ({ files, imageDataList, documentContentList = [], onRemove }) => {
  if (!files || files.length === 0) {
    return null;
  }

  return (
    <div className="flex flex-row overflow-x-auto mx-2 -mt-1 p-2 bg-bolt-elements-background-depth-3 border border-b-none border-bolt-elements-borderColor rounded-lg rounded-b-none">
      {files.map((file, index) => (
        <div key={file.name + file.size} className="mr-2 relative">
          {imageDataList[index] ? (
            <div className="relative">
              <img src={imageDataList[index]} alt={file.name} className="max-h-20 rounded-lg" />
              <button
                onClick={() => onRemove(index)}
                className="absolute -top-1 -right-1 z-10 bg-black rounded-full w-5 h-5 shadow-md hover:bg-gray-900 transition-colors flex items-center justify-center"
              >
                <div className="i-ph:x w-3 h-3 text-gray-200" />
              </button>
              <div className="absolute bottom-0 w-full h-5 flex items-center px-2 rounded-b-lg text-bolt-elements-textTertiary font-thin text-xs bg-bolt-elements-background-depth-2">
                <span className="truncate">{file.name}</span>
              </div>
            </div>
          ) : documentContentList[index] !== undefined ? (
            <div className="relative flex flex-col items-center justify-center w-20 h-20 bg-bolt-elements-background-depth-2 border border-bolt-elements-borderColor rounded-lg overflow-hidden">
              <div className={`${getDocumentIcon(file.name)} w-8 h-8 text-bolt-elements-textSecondary`} />
              <button
                onClick={() => onRemove(index)}
                className="absolute -top-1 -right-1 z-10 bg-black rounded-full w-5 h-5 shadow-md hover:bg-gray-900 transition-colors flex items-center justify-center"
              >
                <div className="i-ph:x w-3 h-3 text-gray-200" />
              </button>
              <div className="absolute bottom-0 w-full h-5 flex items-center px-2 rounded-b-lg text-bolt-elements-textTertiary font-thin text-xs bg-bolt-elements-background-depth-2">
                <span className="truncate">{file.name}</span>
              </div>
            </div>
          ) : null}
        </div>
      ))}
    </div>
  );
};

export default FilePreview;
