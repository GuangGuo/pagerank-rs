use std::{process::exit, path::{PathBuf}};
use clap::Parser;

mod table;
use crate::table::Table;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// enable tracing
    #[arg(short, long)]
    t: bool,

    /// treat graph file as numeric; i.e. input comprises integer vertex names
    #[arg(short, long)]
    n: bool,

    /// the dumping factor
    #[arg(short, long)]
    alpha: f64,

    /// the convergence criterion
    #[arg(short, long)]
    convergence: f64,

    /// hint for internal tables
    #[arg(short, long)]
    size: usize,

    /// delimiter for separating vertex names in each input line
    #[arg(short, long)]
    delim: String,

    /// maximum number of iterations to perform
    #[arg(short, long)]
    max_iterations: usize,

    /// graph_file
    #[arg(short, long, value_name="graph_file")]
    file: PathBuf,
}

fn main() {
    let mut t = Table::new();
    let cli = Cli::parse();

    if cli.t {
        t.set_trace(true);
    }

    if cli.n {
        t.set_numeric(true)
    }

    let alpha = cli.alpha;
    if alpha < 0.0 || alpha >= 1.0 {
        eprintln!("Invalid alpha argument");
        exit(1);
    }
    t.set_alpha(alpha);

    let convergence = cli.convergence;
    if convergence == 0.0 {
        eprintln!("Invalid convergence argument");
        exit(1);
    }
    t.set_convergence(convergence);

    let size = cli.size;
    if size == 0 {
        eprintln!("Invalid size argument");
        exit(1);
    }
    t.set_num_rows(size);

    let iterations = cli.max_iterations;
    if iterations == 0 {
        eprintln!("Invalid iterations argument");
        exit(1);
    }

    let delim = cli.delim;
    t.set_delim(&delim);

    let file = cli.file;
    
    t.print_params();
    println!("Reading input from {} ...", file.display());

    t.read_file(&file).unwrap();

    println!("Calculating pagerank ...");
    t.pagerank();
    println!("Done calculating!");
    t.print_pagerank_v();

}