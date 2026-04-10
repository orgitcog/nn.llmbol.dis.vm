import { BaseProvider, getOpenAILikeModel } from '~/lib/modules/llm/base-provider';
import type { ModelInfo } from '~/lib/modules/llm/types';
import type { IProviderSetting } from '~/types/model';
import type { LanguageModelV1 } from 'ai';

/**
 * IBM Granite Provider
 *
 * Accesses IBM Granite foundation models through IBM watsonx.ai, which exposes
 * an OpenAI-compatible REST API. Requires an IBM Cloud account with watsonx.ai
 * service provisioned.
 *
 * Setup:
 * 1. Create an IBM Cloud account: https://cloud.ibm.com/registration
 * 2. Provision watsonx.ai at: https://dataplatform.cloud.ibm.com/
 * 3. Generate an API key at: https://cloud.ibm.com/iam/apikeys
 * 4. Set WATSONX_PROJECT_ID to your watsonx.ai project ID
 */

const WATSONX_DEFAULT_API_VERSION = '2024-03-14';
export default class GraniteProvider extends BaseProvider {
  name = 'Granite';
  getApiKeyLink = 'https://cloud.ibm.com/iam/apikeys';
  labelForGetApiKey = 'Get IBM Cloud API Key';
  icon = 'i-ph:cube';

  config = {
    apiTokenKey: 'WATSONX_API_KEY',
    baseUrlKey: 'WATSONX_BASE_URL',
    baseUrl: 'https://us-south.ml.cloud.ibm.com/ml/v1',
  };

  staticModels: ModelInfo[] = [
    /*
     * IBM Granite 3.x models - enterprise-grade, open-weight foundation models
     * optimized for coding, reasoning, and business applications.
     */
    {
      name: 'ibm/granite-3-8b-instruct',
      label: 'Granite 3 8B Instruct',
      provider: 'Granite',
      maxTokenAllowed: 8192,
      maxCompletionTokens: 4096,
    },
    {
      name: 'ibm/granite-3-2b-instruct',
      label: 'Granite 3 2B Instruct',
      provider: 'Granite',
      maxTokenAllowed: 8192,
      maxCompletionTokens: 4096,
    },
    {
      name: 'ibm/granite-20b-multilingual',
      label: 'Granite 20B Multilingual',
      provider: 'Granite',
      maxTokenAllowed: 8192,
      maxCompletionTokens: 4096,
    },
    {
      name: 'ibm/granite-34b-code-instruct',
      label: 'Granite 34B Code Instruct',
      provider: 'Granite',
      maxTokenAllowed: 8192,
      maxCompletionTokens: 4096,
    },
    {
      name: 'ibm/granite-8b-code-instruct',
      label: 'Granite 8B Code Instruct',
      provider: 'Granite',
      maxTokenAllowed: 8192,
      maxCompletionTokens: 4096,
    },
  ];

  async getDynamicModels(
    apiKeys?: Record<string, string>,
    settings?: IProviderSetting,
    serverEnv: Record<string, string> = {},
  ): Promise<ModelInfo[]> {
    const { baseUrl, apiKey } = this.getProviderBaseUrlAndKey({
      apiKeys,
      providerSettings: settings,
      serverEnv,
      defaultBaseUrlKey: 'WATSONX_BASE_URL',
      defaultApiTokenKey: 'WATSONX_API_KEY',
    });

    if (!apiKey || !baseUrl) {
      return [];
    }

    const projectId = serverEnv?.WATSONX_PROJECT_ID || process?.env?.WATSONX_PROJECT_ID;

    if (!projectId) {
      return [];
    }

    try {
      const response = await fetch(
        `${baseUrl}/foundation_model_specs?project_id=${encodeURIComponent(projectId)}&filters=function_text_generation`,
        {
          headers: {
            Authorization: `Bearer ${apiKey}`,
            'Content-Type': 'application/json',
          },
        },
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const res = (await response.json()) as any;
      const staticModelIds = this.staticModels.map((m) => m.name);

      return (res.resources || [])
        .filter((model: any) => model.model_id?.startsWith('ibm/granite') && !staticModelIds.includes(model.model_id))
        .map((model: any) => ({
          name: model.model_id,
          label: model.label || model.model_id,
          provider: this.name,
          maxTokenAllowed: model.model_limits?.max_sequence_length || 8192,
          maxCompletionTokens: model.model_limits?.max_output_tokens || 4096,
        }));
    } catch (error) {
      console.log(`${this.name}: Failed to fetch dynamic models`, error);
      return [];
    }
  }

  getModelInstance(options: {
    model: string;
    serverEnv: Env;
    apiKeys?: Record<string, string>;
    providerSettings?: Record<string, IProviderSetting>;
  }): LanguageModelV1 {
    const { model, serverEnv, apiKeys, providerSettings } = options;

    const { baseUrl, apiKey } = this.getProviderBaseUrlAndKey({
      apiKeys,
      providerSettings: providerSettings?.[this.name],
      serverEnv: serverEnv as any,
      defaultBaseUrlKey: 'WATSONX_BASE_URL',
      defaultApiTokenKey: 'WATSONX_API_KEY',
    });

    if (!apiKey) {
      throw new Error(`Missing API key for ${this.name} provider`);
    }

    const projectId =
      providerSettings?.[this.name]?.projectId ||
      (serverEnv as any)?.WATSONX_PROJECT_ID ||
      process?.env?.WATSONX_PROJECT_ID;

    const endpoint = baseUrl || this.config.baseUrl!;

    const apiVersion =
      (providerSettings?.[this.name] as any)?.apiVersion ||
      (serverEnv as any)?.WATSONX_API_VERSION ||
      process?.env?.WATSONX_API_VERSION ||
      WATSONX_DEFAULT_API_VERSION;

    // IBM watsonx.ai exposes an OpenAI-compatible chat completions endpoint
    const watsonxBaseUrl = `${endpoint}/text/chat?version=${encodeURIComponent(apiVersion)}${projectId ? `&project_id=${encodeURIComponent(projectId)}` : ''}`;

    return getOpenAILikeModel(watsonxBaseUrl, apiKey, model);
  }
}
