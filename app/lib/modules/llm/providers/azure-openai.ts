import { BaseProvider } from '~/lib/modules/llm/base-provider';
import type { ModelInfo } from '~/lib/modules/llm/types';
import type { IProviderSetting } from '~/types/model';
import type { LanguageModelV1 } from 'ai';
import { createAzure } from '@ai-sdk/azure';

export default class AzureOpenAIProvider extends BaseProvider {
  name = 'AzureOpenAI';
  getApiKeyLink = 'https://portal.azure.com/#view/Microsoft_Azure_ProjectOxford/CognitiveServicesHub/~/OpenAI';
  labelForGetApiKey = 'Get Azure OpenAI Key';
  icon = 'i-ph:microsoft-teams-logo';

  config = {
    apiTokenKey: 'AZURE_OPENAI_API_KEY',
    baseUrlKey: 'AZURE_OPENAI_RESOURCE_NAME',
  };

  staticModels: ModelInfo[] = [
    /*
     * Azure OpenAI models are accessed through deployment names configured in the Azure portal.
     * These common deployment names align with the underlying model capabilities.
     */
    {
      name: 'gpt-4o',
      label: 'GPT-4o (Azure)',
      provider: 'AzureOpenAI',
      maxTokenAllowed: 128000,
      maxCompletionTokens: 4096,
    },
    {
      name: 'gpt-4o-mini',
      label: 'GPT-4o Mini (Azure)',
      provider: 'AzureOpenAI',
      maxTokenAllowed: 128000,
      maxCompletionTokens: 4096,
    },
    {
      name: 'gpt-4-turbo',
      label: 'GPT-4 Turbo (Azure)',
      provider: 'AzureOpenAI',
      maxTokenAllowed: 128000,
      maxCompletionTokens: 4096,
    },
    {
      name: 'gpt-35-turbo',
      label: 'GPT-3.5 Turbo (Azure)',
      provider: 'AzureOpenAI',
      maxTokenAllowed: 16385,
      maxCompletionTokens: 4096,
    },
  ];

  getModelInstance(options: {
    model: string;
    serverEnv: Env;
    apiKeys?: Record<string, string>;
    providerSettings?: Record<string, IProviderSetting>;
  }): LanguageModelV1 {
    const { model, serverEnv, apiKeys, providerSettings } = options;

    const { apiKey, baseUrl: resourceName } = this.getProviderBaseUrlAndKey({
      apiKeys,
      providerSettings: providerSettings?.[this.name],
      serverEnv: serverEnv as any,
      defaultBaseUrlKey: 'AZURE_OPENAI_RESOURCE_NAME',
      defaultApiTokenKey: 'AZURE_OPENAI_API_KEY',
    });

    if (!apiKey) {
      throw new Error(`Missing API key for ${this.name} provider`);
    }

    if (!resourceName) {
      throw new Error(`Missing resource name for ${this.name} provider. Set AZURE_OPENAI_RESOURCE_NAME.`);
    }

    const apiVersion =
      providerSettings?.[this.name]?.apiVersion ||
      (serverEnv as any)?.AZURE_OPENAI_API_VERSION ||
      process?.env?.AZURE_OPENAI_API_VERSION ||
      '2024-10-01-preview';

    const azure = createAzure({
      resourceName,
      apiKey,
      apiVersion,
    });

    return azure(model);
  }
}
