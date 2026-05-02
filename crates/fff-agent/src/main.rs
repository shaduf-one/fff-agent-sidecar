use std::path::PathBuf;

use clap::{Parser, Subcommand};
use fff_agent::{AgentRequest, AgentResponse, GrepMode, send_request};

#[derive(Debug, Parser)]
#[command(name = "fff-agent", version, about = "Agent-friendly FFF sidecar CLI")]
struct Cli {
    /// Repository path. Defaults to the current git repository.
    #[arg(long, global = true)]
    repo: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Fuzzy file/path search.
    Find {
        query: String,
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    /// Ranked literal content search by default.
    Grep {
        query: String,
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long)]
        regex: bool,
        #[arg(long)]
        fuzzy: bool,
        #[arg(long)]
        json: bool,
    },
    /// Ranked OR search for multiple literal patterns.
    MultiGrep {
        #[arg(long = "pattern", required = true)]
        patterns: Vec<String>,
        #[arg(long)]
        constraints: Option<String>,
        #[arg(long, default_value_t = 30)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    /// Show daemon/index status.
    Status,
    /// Force a synchronous rescan.
    Rescan,
    /// Stop the per-repository daemon.
    Stop,
    /// Internal daemon process.
    #[command(hide = true)]
    Daemon {
        #[arg(long)]
        repo: PathBuf,
        #[arg(long)]
        socket: PathBuf,
    },
}

fn main() {
    if let Err(error) = run() {
        eprintln!("fff-agent: {error}");
        std::process::exit(1);
    }
}

fn run() -> fff_agent::Result<()> {
    let cli = Cli::parse();
    if let Command::Daemon { repo, socket } = cli.command {
        return fff_agent::run_daemon(&repo, &socket);
    }

    let repo = cli
        .repo
        .unwrap_or(std::env::current_dir().map_err(fff_agent::AgentError::CurrentDir)?);
    let request = match cli.command {
        Command::Find { query, limit, json } => AgentRequest::Find { query, limit, json },
        Command::Grep {
            query,
            limit,
            regex,
            fuzzy,
            json,
        } => AgentRequest::Grep {
            query,
            limit,
            mode: if fuzzy {
                GrepMode::Fuzzy
            } else if regex {
                GrepMode::Regex
            } else {
                GrepMode::Plain
            },
            json,
        },
        Command::MultiGrep {
            patterns,
            constraints,
            limit,
            json,
        } => AgentRequest::MultiGrep {
            patterns,
            constraints,
            limit,
            json,
        },
        Command::Status => AgentRequest::Status,
        Command::Rescan => AgentRequest::Rescan,
        Command::Stop => AgentRequest::Shutdown,
        Command::Daemon { .. } => unreachable!("daemon command handled above"),
    };

    let response = send_request(&repo, request)?;
    print_response(response);
    Ok(())
}

fn print_response(response: AgentResponse) {
    match response {
        AgentResponse::Text { text } => println!("{text}"),
        AgentResponse::Status {
            repo,
            indexed_files,
            scanning,
            watch,
        } => {
            println!("repo: {repo}");
            println!("indexed_files: {indexed_files}");
            println!("scanning: {scanning}");
            println!("watch: {watch}");
        }
        AgentResponse::Error { message } => {
            eprintln!("{message}");
            std::process::exit(1);
        }
        AgentResponse::Shutdown => println!("stopped"),
    }
}
