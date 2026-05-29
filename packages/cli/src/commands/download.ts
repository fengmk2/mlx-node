import { setToken } from './hf-token.js';

function printHelp(): void {
  console.log(`
Download models and datasets from HuggingFace

Usage:
  mlx download model      Download a model from HuggingFace
  mlx download dataset    Download a dataset from HuggingFace

Options:
  --set-token             Set your HuggingFace token (saved to the OS keychain;
                          used by both model and dataset downloads)
  -h, --help              Show this help message

Run mlx download <subcommand> --help for more information.
`);
}

/**
 * Dispatcher for `mlx download`.
 *
 * The HuggingFace token is a download-wide credential, so download-level
 * flags (`--set-token`, `--help`) are handled here rather than being gated
 * behind a `model`/`dataset` subcommand. The top-level CLI delegates the
 * whole argument list to this function instead of pre-parsing it.
 */
export async function run(argv: string[]): Promise<void> {
  const subcommand = argv[0];

  if (!subcommand || subcommand === '--help' || subcommand === '-h') {
    printHelp();
    return;
  }

  // Download-wide flags, valid with or without a subcommand.
  if (subcommand === '--set-token') {
    await setToken();
    return;
  }

  const rest = argv.slice(1);

  if (subcommand === 'model') {
    const { run: runModel } = await import('./download-model.js');
    await runModel(rest);
  } else if (subcommand === 'dataset') {
    const { run: runDataset } = await import('./download-dataset.js');
    await runDataset(rest);
  } else {
    console.error(`Unknown download subcommand: ${subcommand}`);
    console.error('Available: model, dataset');
    process.exit(1);
  }
}
