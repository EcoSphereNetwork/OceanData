import { Command } from 'commander';
import { publishDataset, fetchResults, getMarketplaceListings } from './src/index';

const program = new Command();
program
  .name('oceandata')
  .description('OceanData TypeScript SDK CLI');

program
  .command('list')
  .description('List marketplace entries')
  .action(async () => {
    const listings = await getMarketplaceListings();
    console.log(JSON.stringify(listings));
  });

program
  .command('results <id>')
  .description('Fetch job results')
  .action(async (id) => {
    const res = await fetchResults(id);
    console.log(JSON.stringify(res));
  });

program
  .command('publish <name>')
  .option('-p, --price <price>', 'Price', '0')
  .description('Publish dataset')
  .action(async (name, options) => {
    const res = await publishDataset(name, { metadata: { title: name }, price: parseFloat(options.price) });
    console.log(JSON.stringify(res));
  });

program.parseAsync(process.argv);

