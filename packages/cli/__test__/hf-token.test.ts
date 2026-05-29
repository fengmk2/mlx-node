/**
 * Unit tests for the shared HuggingFace token module.
 *
 * Keychain access is routed through the exported `keychain` object so these
 * tests can spy on `read`/`write` without touching the real OS keychain, and
 * `setToken` accepts an injectable IO so no interactive prompt runs.
 */
import { describe, it, expect, vi, afterEach } from 'vite-plus/test';

import { resolveHuggingFaceToken, setToken, keychain } from '../src/commands/hf-token.js';

afterEach(() => {
  vi.restoreAllMocks();
  delete process.env.HUGGINGFACE_TOKEN;
});

describe('resolveHuggingFaceToken', () => {
  it('prefers the keychain token over the environment variable', async () => {
    vi.spyOn(keychain, 'read').mockResolvedValue('hf_from_keychain');
    process.env.HUGGINGFACE_TOKEN = 'hf_from_env';
    expect(await resolveHuggingFaceToken()).toBe('hf_from_keychain');
  });

  it('falls back to HUGGINGFACE_TOKEN when the keychain is empty', async () => {
    vi.spyOn(keychain, 'read').mockResolvedValue(undefined);
    process.env.HUGGINGFACE_TOKEN = 'hf_from_env';
    expect(await resolveHuggingFaceToken()).toBe('hf_from_env');
  });

  it('returns undefined when neither source has a token', async () => {
    vi.spyOn(keychain, 'read').mockResolvedValue(undefined);
    delete process.env.HUGGINGFACE_TOKEN;
    expect(await resolveHuggingFaceToken()).toBeUndefined();
  });
});

describe('setToken', () => {
  it('persists the entered token to the keychain', async () => {
    const writeSpy = vi.spyOn(keychain, 'write').mockResolvedValue(undefined);
    await setToken({ readToken: async () => 'hf_entered', log: () => {} });
    expect(writeSpy).toHaveBeenCalledWith('hf_entered');
  });

  it('does not write when an empty token is entered', async () => {
    const writeSpy = vi.spyOn(keychain, 'write').mockResolvedValue(undefined);
    await setToken({ readToken: async () => '', log: () => {} });
    expect(writeSpy).not.toHaveBeenCalled();
  });
});
