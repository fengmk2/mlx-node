/**
 * Routing tests for the `mlx download` dispatcher.
 *
 * The regression these guard against: `mlx download --set-token` used to be
 * rejected as `Unknown download subcommand: --set-token` because the token
 * flag lived only inside the `download model` subcommand and the top-level
 * dispatcher treated `args[1]` as a subcommand unconditionally. The token is
 * a download-wide credential, so `--set-token` must be handled at the
 * `download` level, with or without a model/dataset subcommand.
 *
 * The shared token module and the lazily-imported subcommand modules are
 * mocked so routing is exercised without prompting or hitting the network.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vite-plus/test';

vi.mock('../src/commands/hf-token.js', () => ({
  setToken: vi.fn(async () => {}),
  resolveHuggingFaceToken: vi.fn(async () => undefined),
}));
vi.mock('../src/commands/download-model.js', () => ({ run: vi.fn(async () => {}) }));
vi.mock('../src/commands/download-dataset.js', () => ({ run: vi.fn(async () => {}) }));

import { run as runDataset } from '../src/commands/download-dataset.js';
import { run as runModel } from '../src/commands/download-model.js';
import { run as runDownload } from '../src/commands/download.js';
import { setToken } from '../src/commands/hf-token.js';

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('mlx download dispatcher', () => {
  it('handles `--set-token` at the download level instead of rejecting it as a subcommand', async () => {
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    await runDownload(['--set-token']);

    expect(setToken).toHaveBeenCalledTimes(1);
    for (const call of errSpy.mock.calls) {
      expect(String(call[0])).not.toContain('Unknown download subcommand');
    }
  });

  it('routes the `model` subcommand to the model downloader with the remaining args', async () => {
    await runDownload(['model', '-m', 'foo/bar']);
    expect(runModel).toHaveBeenCalledWith(['-m', 'foo/bar']);
  });

  it('routes the `dataset` subcommand to the dataset downloader with the remaining args', async () => {
    await runDownload(['dataset', '-d', 'foo/bar']);
    expect(runDataset).toHaveBeenCalledWith(['-d', 'foo/bar']);
  });

  it('prints help (mentioning the subcommands) when no subcommand is given', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    await runDownload([]);
    const output = logSpy.mock.calls.map((c) => String(c[0])).join('\n');
    expect(output).toContain('mlx download model');
    expect(output).toContain('mlx download dataset');
  });

  it('errors and exits on an unknown subcommand', async () => {
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation((() => undefined) as never);

    await runDownload(['bogus']);

    const errors = errSpy.mock.calls.map((c) => String(c[0])).join('\n');
    expect(errors).toContain('Unknown download subcommand: bogus');
    expect(exitSpy).toHaveBeenCalledWith(1);
  });
});
