import { describe, expect, it, beforeEach, vi } from 'vitest';
import { LLMManager } from '../llm/manager';
import { BaseProvider } from '../llm/base-provider';
import type { ModelInfo, ProviderConfig } from '../llm/types';
import type { LanguageModelV1 } from 'ai';

// ---------------------------------------------------------------------------
// Minimal concrete provider for unit testing
// ---------------------------------------------------------------------------

class MockProvider extends BaseProvider {
  name = 'MockProvider';
  getApiKeyLink = 'https://example.com/api-keys';

  config: ProviderConfig = {
    apiTokenKey: 'MOCK_API_KEY',
    baseUrlKey: 'MOCK_BASE_URL',
    baseUrl: 'https://api.mock.example.com',
  };

  staticModels: ModelInfo[] = [
    {
      name: 'mock-model-small',
      label: 'Mock Model Small',
      provider: 'MockProvider',
      maxTokenAllowed: 8192,
      maxCompletionTokens: 1024,
    },
    {
      name: 'mock-model-large',
      label: 'Mock Model Large',
      provider: 'MockProvider',
      maxTokenAllowed: 128000,
      maxCompletionTokens: 16384,
    },
  ];

  getModelInstance(): LanguageModelV1 {
    return {} as LanguageModelV1;
  }
}

class MockProviderWithDynamic extends BaseProvider {
  name = 'MockProviderDynamic';

  config: ProviderConfig = {
    apiTokenKey: 'MOCK_DYNAMIC_API_KEY',
  };

  staticModels: ModelInfo[] = [
    {
      name: 'static-model',
      label: 'Static Model',
      provider: 'MockProviderDynamic',
      maxTokenAllowed: 4096,
    },
  ];

  async getDynamicModels(): Promise<ModelInfo[]> {
    return [
      {
        name: 'dynamic-model-1',
        label: 'Dynamic Model 1',
        provider: 'MockProviderDynamic',
        maxTokenAllowed: 32000,
      },
      {
        name: 'dynamic-model-2',
        label: 'Dynamic Model 2',
        provider: 'MockProviderDynamic',
        maxTokenAllowed: 64000,
      },
    ];
  }

  getModelInstance(): LanguageModelV1 {
    return {} as LanguageModelV1;
  }
}

// ---------------------------------------------------------------------------
// BaseProvider unit tests
// ---------------------------------------------------------------------------

describe('BaseProvider', () => {
  let provider: MockProvider;

  beforeEach(() => {
    provider = new MockProvider();
  });

  describe('Provider metadata', () => {
    it('should expose a name', () => {
      expect(provider.name).toBe('MockProvider');
    });

    it('should expose static models', () => {
      expect(provider.staticModels).toHaveLength(2);
      expect(provider.staticModels[0].name).toBe('mock-model-small');
    });

    it('should expose a config with apiTokenKey', () => {
      expect(provider.config.apiTokenKey).toBe('MOCK_API_KEY');
    });
  });

  describe('getProviderBaseUrlAndKey', () => {
    it('should return baseUrl from config when no overrides are given', () => {
      const result = provider.getProviderBaseUrlAndKey({
        defaultBaseUrlKey: 'MOCK_BASE_URL',
        defaultApiTokenKey: 'MOCK_API_KEY',
      });
      expect(result.baseUrl).toBe('https://api.mock.example.com');
    });

    it('should prefer providerSettings baseUrl over config default', () => {
      const result = provider.getProviderBaseUrlAndKey({
        providerSettings: { baseUrl: 'https://custom.example.com' },
        defaultBaseUrlKey: 'MOCK_BASE_URL',
        defaultApiTokenKey: 'MOCK_API_KEY',
      });
      expect(result.baseUrl).toBe('https://custom.example.com');
    });

    it('should strip trailing slash from baseUrl', () => {
      const result = provider.getProviderBaseUrlAndKey({
        providerSettings: { baseUrl: 'https://custom.example.com/' },
        defaultBaseUrlKey: 'MOCK_BASE_URL',
        defaultApiTokenKey: 'MOCK_API_KEY',
      });
      expect(result.baseUrl).toBe('https://custom.example.com');
    });

    it('should prefer apiKeys over serverEnv for api key resolution', () => {
      const result = provider.getProviderBaseUrlAndKey({
        apiKeys: { MockProvider: 'key-from-api-keys' },
        serverEnv: { MOCK_API_KEY: 'key-from-server-env' },
        defaultBaseUrlKey: 'MOCK_BASE_URL',
        defaultApiTokenKey: 'MOCK_API_KEY',
      });
      expect(result.apiKey).toBe('key-from-api-keys');
    });

    it('should fall back to serverEnv api key when apiKeys not provided', () => {
      const result = provider.getProviderBaseUrlAndKey({
        serverEnv: { MOCK_API_KEY: 'key-from-server-env' },
        defaultBaseUrlKey: 'MOCK_BASE_URL',
        defaultApiTokenKey: 'MOCK_API_KEY',
      });
      expect(result.apiKey).toBe('key-from-server-env');
    });

    it('should return undefined apiKey when none is available', () => {
      const result = provider.getProviderBaseUrlAndKey({
        defaultBaseUrlKey: 'MOCK_BASE_URL',
        defaultApiTokenKey: 'MOCK_API_KEY',
      });
      expect(result.apiKey).toBeUndefined();
    });
  });

  describe('Dynamic model caching', () => {
    it('should return null from cache when no models have been stored', () => {
      const result = provider.getModelsFromCache({});
      expect(result).toBeNull();
    });

    it('should store and retrieve dynamic models from cache', () => {
      const options = { apiKeys: { MockProvider: 'test-key' } };
      const models: ModelInfo[] = [
        { name: 'cached-model', label: 'Cached', provider: 'MockProvider', maxTokenAllowed: 4096 },
      ];

      provider.storeDynamicModels(options, models);
      const cached = provider.getModelsFromCache(options);

      expect(cached).toHaveLength(1);
      expect(cached![0].name).toBe('cached-model');
    });

    it('should invalidate cache when options change', () => {
      const options1 = { apiKeys: { MockProvider: 'key-1' } };
      const options2 = { apiKeys: { MockProvider: 'key-2' } };
      const models: ModelInfo[] = [
        { name: 'cached-model', label: 'Cached', provider: 'MockProvider', maxTokenAllowed: 4096 },
      ];

      provider.storeDynamicModels(options1, models);
      const result = provider.getModelsFromCache(options2);

      expect(result).toBeNull();
    });
  });
});

// ---------------------------------------------------------------------------
// LLMManager unit tests
// ---------------------------------------------------------------------------

describe('LLMManager', () => {
  // Reset the singleton between tests using a fresh instance trick
  let manager: LLMManager;

  beforeEach(() => {
    // Access private static to reset singleton for test isolation
    (LLMManager as any)._instance = undefined;
    manager = LLMManager.getInstance();
  });

  describe('Singleton behaviour', () => {
    it('should return the same instance on repeated calls', () => {
      const m1 = LLMManager.getInstance();
      const m2 = LLMManager.getInstance();
      expect(m1).toBe(m2);
    });

    it('should expose env passed at construction', () => {
      (LLMManager as any)._instance = undefined;
      const m = LLMManager.getInstance({ CUSTOM_VAR: 'hello' });
      expect(m.env.CUSTOM_VAR).toBe('hello');
    });
  });

  describe('Provider registration', () => {
    it('should register a provider', () => {
      const provider = new MockProvider();
      manager.registerProvider(provider);

      const retrieved = manager.getProvider('MockProvider');
      expect(retrieved).toBe(provider);
    });

    it('should not register the same provider twice', () => {
      const provider = new MockProvider();
      manager.registerProvider(provider);
      manager.registerProvider(provider); // second call should be silently skipped

      // Provider list should still contain exactly one MockProvider
      const providers = manager.getAllProviders().filter((p) => p.name === 'MockProvider');
      expect(providers).toHaveLength(1);
    });

    it('should return undefined for an unknown provider', () => {
      expect(manager.getProvider('NonExistent')).toBeUndefined();
    });
  });

  describe('Model listing', () => {
    it('should include static models from registered providers', () => {
      const provider = new MockProvider();
      manager.registerProvider(provider);

      const models = manager.getModelList();
      const names = models.map((m) => m.name);

      expect(names).toContain('mock-model-small');
      expect(names).toContain('mock-model-large');
    });

    it('should return static models via getStaticModelList', () => {
      const provider = new MockProvider();
      manager.registerProvider(provider);

      const staticModels = manager.getStaticModelList();
      expect(staticModels.length).toBeGreaterThanOrEqual(2);
    });

    it('should return static models for a specific provider', () => {
      const provider = new MockProvider();
      manager.registerProvider(provider);

      const models = manager.getStaticModelListFromProvider(provider);
      expect(models).toHaveLength(2);
    });

    it('should throw when requesting static models for an unregistered provider', () => {
      const unregistered = new MockProvider();
      expect(() => manager.getStaticModelListFromProvider(unregistered)).toThrow();
    });
  });

  describe('Default provider', () => {
    it('should return the first registered provider as default', () => {
      const provider = new MockProvider();
      manager.registerProvider(provider);

      const def = manager.getDefaultProvider();
      expect(def.name).toBe('MockProvider');
    });

    it('should throw when no providers are registered', () => {
      expect(() => manager.getDefaultProvider()).toThrow();
    });
  });

  describe('getAllProviders', () => {
    it('should return all registered providers', () => {
      manager.registerProvider(new MockProvider());
      manager.registerProvider(new MockProviderWithDynamic());

      const all = manager.getAllProviders();
      const names = all.map((p) => p.name);

      expect(names).toContain('MockProvider');
      expect(names).toContain('MockProviderDynamic');
    });
  });

  describe('updateModelList', () => {
    it('should merge static and dynamic models', async () => {
      const provider = new MockProviderWithDynamic();
      manager.registerProvider(provider);

      const models = await manager.updateModelList({});
      const names = models.map((m) => m.name);

      expect(names).toContain('static-model');
      expect(names).toContain('dynamic-model-1');
      expect(names).toContain('dynamic-model-2');
    });

    it('should de-duplicate dynamic models that share a name with static models', async () => {
      // Create a provider whose dynamic list overlaps with its static list
      class OverlapProvider extends BaseProvider {
        name = 'OverlapProvider';
        config: ProviderConfig = { apiTokenKey: 'OVERLAP_KEY' };
        staticModels: ModelInfo[] = [
          { name: 'shared-model', label: 'Shared', provider: 'OverlapProvider', maxTokenAllowed: 8192 },
        ];
        async getDynamicModels(): Promise<ModelInfo[]> {
          return [
            { name: 'shared-model', label: 'Shared Dynamic', provider: 'OverlapProvider', maxTokenAllowed: 16384 },
            { name: 'unique-dynamic', label: 'Unique Dynamic', provider: 'OverlapProvider', maxTokenAllowed: 32768 },
          ];
        }
        getModelInstance(): LanguageModelV1 {
          return {} as LanguageModelV1;
        }
      }

      manager.registerProvider(new OverlapProvider());
      const models = await manager.updateModelList({});
      const sharedModels = models.filter((m) => m.name === 'shared-model' && m.provider === 'OverlapProvider');

      // Dynamic version should win; static duplicate should be removed
      expect(sharedModels).toHaveLength(1);
      expect(sharedModels[0].maxTokenAllowed).toBe(16384);
    });

    it('should filter out disabled providers when providerSettings are given', async () => {
      manager.registerProvider(new MockProvider());
      manager.registerProvider(new MockProviderWithDynamic());

      const models = await manager.updateModelList({
        providerSettings: {
          MockProvider: { enabled: true },
          MockProviderDynamic: { enabled: false },
        },
      });

      const names = models.map((m) => m.name);
      expect(names).toContain('mock-model-small');
      expect(names).not.toContain('dynamic-model-1');
    });

    it('should gracefully handle dynamic model fetch errors', async () => {
      class FailingProvider extends BaseProvider {
        name = 'FailingProvider';
        config: ProviderConfig = { apiTokenKey: 'FAIL_KEY' };
        staticModels: ModelInfo[] = [
          { name: 'fail-static', label: 'Fail Static', provider: 'FailingProvider', maxTokenAllowed: 4096 },
        ];
        async getDynamicModels(): Promise<ModelInfo[]> {
          throw new Error('Network error');
        }
        getModelInstance(): LanguageModelV1 {
          return {} as LanguageModelV1;
        }
      }

      manager.registerProvider(new FailingProvider());

      // Should not throw; static models should still be returned
      const models = await manager.updateModelList({});
      const names = models.map((m) => m.name);
      expect(names).toContain('fail-static');
    });
  });

  describe('getModelListFromProvider', () => {
    it('should return static models for a provider without dynamic support', async () => {
      const provider = new MockProvider();
      manager.registerProvider(provider);

      const models = await manager.getModelListFromProvider(provider, {});
      expect(models).toHaveLength(2);
    });

    it('should return combined models for a provider with dynamic support', async () => {
      const provider = new MockProviderWithDynamic();
      manager.registerProvider(provider);

      const models = await manager.getModelListFromProvider(provider, {});
      const names = models.map((m) => m.name);

      expect(names).toContain('static-model');
      expect(names).toContain('dynamic-model-1');
    });

    it('should use cached dynamic models when available', async () => {
      const provider = new MockProviderWithDynamic();
      manager.registerProvider(provider);

      const getDynamicSpy = vi.spyOn(provider, 'getDynamicModels');

      // First call populates cache
      await manager.getModelListFromProvider(provider, {});
      // Second call should use cache
      await manager.getModelListFromProvider(provider, {});

      expect(getDynamicSpy).toHaveBeenCalledTimes(1);
    });

    it('should throw when the provider is not registered', async () => {
      const unregistered = new MockProvider();
      await expect(manager.getModelListFromProvider(unregistered, {})).rejects.toThrow();
    });
  });
});
