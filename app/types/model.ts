import type { ModelInfo } from '~/lib/modules/llm/types';

export type ProviderInfo = {
  staticModels: ModelInfo[];
  name: string;
  getDynamicModels?: (
    providerName: string,
    apiKeys?: Record<string, string>,
    providerSettings?: IProviderSetting,
    serverEnv?: Record<string, string>,
  ) => Promise<ModelInfo[]>;
  getApiKeyLink?: string;
  labelForGetApiKey?: string;
  icon?: string;
};

export interface IProviderSetting {
  enabled?: boolean;
  baseUrl?: string;
  OPENAI_LIKE_API_MODELS?: string;

  /** Azure OpenAI API version (e.g. 2024-10-01-preview) */
  apiVersion?: string;

  /** IBM watsonx project ID */
  projectId?: string;
}

export type IProviderConfig = ProviderInfo & {
  settings: IProviderSetting;
};
