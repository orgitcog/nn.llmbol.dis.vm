import { BaseProvider } from '~/lib/modules/llm/base-provider';
import type { ModelInfo } from '~/lib/modules/llm/types';
import type { IProviderSetting } from '~/types/model';
import type { LanguageModelV1 } from 'ai';
import { createVertex } from '@ai-sdk/google-vertex';

export default class VertexAIProvider extends BaseProvider {
  name = 'VertexAI';
  getApiKeyLink = 'https://console.cloud.google.com/vertex-ai';
  labelForGetApiKey = 'Open Vertex AI Console';
  icon = 'i-ph:google-logo';

  config = {
    apiTokenKey: 'GOOGLE_APPLICATION_CREDENTIALS',
    baseUrlKey: 'VERTEX_AI_PROJECT',
  };

  staticModels: ModelInfo[] = [
    /*
     * Vertex AI Gemini models - same underlying models as Google AI Studio
     * but accessed through Google Cloud with enterprise-grade features.
     */
    {
      name: 'gemini-1.5-pro',
      label: 'Gemini 1.5 Pro (Vertex)',
      provider: 'VertexAI',
      maxTokenAllowed: 2000000,
      maxCompletionTokens: 8192,
    },
    {
      name: 'gemini-1.5-flash',
      label: 'Gemini 1.5 Flash (Vertex)',
      provider: 'VertexAI',
      maxTokenAllowed: 1000000,
      maxCompletionTokens: 8192,
    },
    {
      name: 'gemini-2.0-flash-001',
      label: 'Gemini 2.0 Flash (Vertex)',
      provider: 'VertexAI',
      maxTokenAllowed: 1000000,
      maxCompletionTokens: 8192,
    },
    {
      name: 'gemini-2.0-pro-exp-02-05',
      label: 'Gemini 2.0 Pro Exp (Vertex)',
      provider: 'VertexAI',
      maxTokenAllowed: 2000000,
      maxCompletionTokens: 8192,
    },
  ];

  getModelInstance(options: {
    model: string;
    serverEnv: Env;
    apiKeys?: Record<string, string>;
    providerSettings?: Record<string, IProviderSetting>;
  }): LanguageModelV1 {
    const { model, serverEnv, apiKeys, providerSettings } = options;

    const { baseUrl: project } = this.getProviderBaseUrlAndKey({
      apiKeys,
      providerSettings: providerSettings?.[this.name],
      serverEnv: serverEnv as any,
      defaultBaseUrlKey: 'VERTEX_AI_PROJECT',
      defaultApiTokenKey: 'GOOGLE_APPLICATION_CREDENTIALS',
    });

    const location =
      (providerSettings?.[this.name] as any)?.location ||
      (serverEnv as any)?.VERTEX_AI_LOCATION ||
      process?.env?.VERTEX_AI_LOCATION ||
      'us-central1';

    if (!project) {
      throw new Error(`Missing project ID for ${this.name} provider. Set VERTEX_AI_PROJECT.`);
    }

    const vertex = createVertex({
      project,
      location,
    });

    return vertex(model);
  }
}
