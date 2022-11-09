use std::{collections::HashMap, process::exit, fs::File, io::{self, BufReader, BufRead}, path::PathBuf};

const DEFAULT_ALPHA: f64 = 0.85;
// convergence 收敛性
const DEFAULT_CONVERGENCE: f64 = 0.00001;
const DEFAULT_MAX_ITERATIONS: usize = 10000;
const DEFAULT_NUMERIC: bool = false;
const DEFAULT_DELIM: &str = " => ";

/// A PageRank calculator. It is responsible for reading data, performing 
/// the algorithmic calculations, and outputing the results.
pub struct Table {
    trace: bool,  // enabling tracing output
    alpha: f64,  // the pagerank damping factor 阻尼系数
    convergence: f64,
    max_iterations: usize,
    delim: String,
    numeric: bool,  // input graph has numeric, zero-based indexed vertices
    num_outgoing: Vec<usize>,  // number of outgoing links per column
    rows: Vec<Vec<usize>>,  // the rowns of the hyperlink matrix
    nodes_to_idx: HashMap<String, usize>,  // mapping from string node IDs to numeric
    idx_to_nodes: HashMap<usize, String>,  // mapping from numeric node IDs to string
    pr: Vec<f64>,  // the pagerank table
}

impl Default for Table {
    fn default() -> Self {
        Self { 
            trace: false, 
            alpha: DEFAULT_ALPHA, 
            convergence: DEFAULT_CONVERGENCE, 
            max_iterations: DEFAULT_MAX_ITERATIONS, 
            delim: DEFAULT_DELIM.to_string(), 
            numeric: DEFAULT_NUMERIC, 
            num_outgoing: Vec::new(), 
            rows: Vec::new(), 
            nodes_to_idx: HashMap::new(), 
            idx_to_nodes: HashMap::new(), 
            pr: Vec::new(), 
        }
    }
}

impl Table {
    fn insert_into_vector<T>(v: &mut Vec<T>, t: T) -> bool 
        where T: PartialOrd
    {
        let mut i = 0;
        // 除了加 .iter() 有更符合直觉的办法吗?
        for item in v.iter() {
            if *item > t {
                break;
            }
            i += 1;
        }
        
        if i == v.len() {
            v.push(t);
            true
        } else {
            v.insert(i, t);
            false
        }
    }

    /// Clears all internal data structures so that the table can be used 
    /// for new input and calculations.
    fn reset(&mut self) {
        self.num_outgoing.clear();
        self.rows.clear();
        self.nodes_to_idx.clear();
        self.idx_to_nodes.clear();
        self.pr.clear();
    }

    /// Adds a mapping from a node string ID (key) to a numeric one to the 
    /// internal mapping tables.
    /// 
    /// Returns the mapped value of the node; if the node has already 
    /// been mapped, the already mapped index.
    fn insert_mapping(&mut self, key: String) -> usize {
        match self.nodes_to_idx.get(&key) {
            Some(&index) => index,
            None => {
                let idx = self.nodes_to_idx.len();
                self.nodes_to_idx.insert(key.clone(), idx);
                self.idx_to_nodes.insert(idx, key.clone());
                idx
            }
        }
    }

    /// Adds an arc to the hyperlink matrix between from and to.
    fn add_arc(&mut self, from: usize, to: usize) -> bool {
        let mut ret = false;
        let mut max_dim = if from > to {
            from
        } else {
            to
        };

        if self.trace {
            println!("checking to add {} => {}", from, to);
        }

        if self.rows.len() <= max_dim {
            max_dim += 1;
            if self.trace {
                println!("resizing rows from {} to {}", self.rows.len(), max_dim);
            }
            
            self.rows.resize_with(max_dim, || Vec::new());
            if self.num_outgoing.len() <= max_dim {
                self.num_outgoing.resize(max_dim, 0);
            }
        }

        ret = Self::insert_into_vector(&mut self.rows[to], from);
        
        if ret {
            self.num_outgoing[from] += 1;
            if self.trace {
                println!("added {} => {}", from, to);
            }
        }

        ret
    }

    pub fn new() -> Table {
        Default::default()
    }

    /// Reserves space for the internal tables used for the PageRank calculation.
    /// It is not necessory to call the method; space will be reserved as needed;
    /// however, if the size of the internal tables is known beforehand and is 
    /// used to initialize them, all space will be allocated at the method call 
    /// (instead of during calculations) resulting in faster operation.
    /// 
    /// The size parameter passed refers to the number of rows of the link 
    /// matrix.
    pub fn reserve(&mut self, size: usize) {
        self.num_outgoing.reserve(size);
        self.rows.reserve(size);
    }

    /// Returns the number of rows of the link matrix.
    pub fn get_num_rows(&self) -> usize {
        self.rows.len()
    }

    /// Sets the number of rows of the link matrix.
    pub fn set_num_rows(&mut self, num_rows: usize) {
        self.num_outgoing.resize(num_rows, 0);
        self.rows.resize_with(num_rows, || { Vec::new() });
    }

    /// Reads the graph described in filename.
    pub fn read_file(&mut self, filename: &PathBuf) -> io::Result<i32> {
        self.reset();

        let file = File::open(filename)?;
        let infile = BufReader::new(file);
        let delim_len = self.delim.len();
        let mut linenum = 0;
        for line_result in infile.lines() {
            let line = line_result?;
            let mut from: &str;
            let mut to: &str;
            let from_idx: usize;
            let to_idx: usize;
            let pos = line.find(self.delim.as_str());
            
            if pos.is_some() {
                let pos = pos.unwrap();
                from = &line[0..pos];
                from = from.trim();
                if !self.numeric {
                    from_idx = self.insert_mapping(from.to_string());
                } else {
                    from_idx = from.parse().unwrap();
                }

                to = &line[pos+delim_len..];
                to = to.trim();
                if !self.numeric {
                    to_idx = self.insert_mapping(to.to_string());
                } else {
                    to_idx = to.parse().unwrap();
                }
                self.add_arc(from_idx, to_idx);
            }

            linenum += 1;
            if linenum != 0 && linenum % 100000 == 0 {
                println!("read {} lines, {} vertices", linenum, self.rows.len());
            }
        }

        println!("read {} lines, {} vertices", linenum, self.rows.len());

        self.nodes_to_idx.clear();
        self.reserve(self.idx_to_nodes.len());

        Ok(0)
    }

    /// Calculates the pagerank of the hyperlink matrix.
    pub fn pagerank(&mut self) {
        let mut diff: f64 = 1.0;
        let mut sum_pr: f64;  // sum of current pagerank vector elements
        let mut dangling_pr: f64;  // sum of current pagerank vector elements for dangling nodes
        let mut num_iterations = 0;
        let mut old_pr: Vec<f64> = Vec::new();

        let num_rows = self.rows.len();

        if num_rows == 0 {
            return;
        }

        self.pr.resize(num_rows, 0.0);

        self.pr[0] = 1.0;

        if self.trace {
            self.print_pagerank();
        }

        while diff > self.convergence && num_iterations < self.max_iterations {
            sum_pr = 0.0;
            dangling_pr = 0.0;

            for (k, cpr) in self.pr.iter().enumerate() {
                sum_pr += cpr;
                if self.num_outgoing[k] == 0 {
                    dangling_pr += cpr;
                }
            }

            if num_iterations == 0 {
                old_pr = self.pr.clone();
            } else {
                // Normalize so that we start with sum equal to one
                let mut i = 0;
                while i < self.pr.len() {
                    old_pr[i] = self.pr[i] / sum_pr;
                    i += 1;
                }
            }

            // After normalisation the elements of the pagerank vector sum to one
            sum_pr = 1.0;

            // An element of the A x I vector; all elements are identical
            let one_Av = self.alpha * dangling_pr / num_rows as f64;

            // An element of the 1 x I vector; all elements are identical
            let one_Iv = (1.0 - self.alpha) * sum_pr / num_rows as f64;

            // The difference to be checked for convergence
            diff = 0.0;
            let mut i = 0;
            while i < num_rows {
                // The corresponding element of the H multiplication
                let mut h = 0.0;
                for ci in &self.rows[i] {
                    let h_v = if self.num_outgoing[*ci] != 0 {
                        1.0 / self.num_outgoing[*ci] as f64
                    } else {
                        0.0
                    };
                    if num_iterations == 0 && self.trace {
                        println!("h[{},{}]={}", i, ci, h_v);
                    }

                    h += h_v * old_pr[*ci];
                }
                h *= self.alpha;
                self.pr[i] = h + one_Av + one_Iv;
                let abs = if self.pr[i] > old_pr[i] {
                    self.pr[i] - old_pr[i]
                } else {
                    old_pr[i] - self.pr[i]
                };
                diff += abs;

                i += 1;
            }

            num_iterations += 1;
            if self.trace {
                print!("{}: ", num_iterations);
                self.print_pagerank();
            }
        }
    }

    /// Returns the pagerank vector of the hyperlink matrix.
    pub fn get_pagerank(&self) -> &Vec<f64> {
        &self.pr
    }

    /// Returns the name of the node with the given index. If the nodes are 
    /// numeric the name is the string representation of the number. if the 
    /// nodes are not numeric, the name is the original node name as it was 
    /// input from read_file(&str)
    pub fn get_node_name(&self, index: usize) -> String {
        if self.numeric {
            index.to_string()
        } else {
            self.idx_to_nodes[&index].to_string()
        }
    }

    pub fn get_mapping(&self) -> &HashMap<usize, String> {
        &self.idx_to_nodes
    }

    /// Returns the pagerank damping factor.
    pub fn get_alpha(&self) -> f64 {
        self.alpha
    }

    /// Sets the pagerank damping factor.
    pub fn set_alpha(&mut self, a: f64) {
        self.alpha = a;
    }

    /// Returns the maximum number of iterations that the pagerank algorithm 
    /// will perform.
    pub fn get_max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Sets the maximum number of iterations that the pagerank algorithm 
    /// will perform.
    pub fn set_max_iterations(&mut self, i: usize) {
        self.max_iterations = i;
    }

    /// Returns the value that is used to determine convergence of the
    /// pagerank calculation algorithm.
    pub fn get_convergence(&self) -> f64 {
        self.convergence
    }

    /// Sets the value that is used to determine convergence of the
    /// pagerank calculation algorithm.
    pub fn set_convergence(&mut self, c: f64) {
        self.convergence = c;
    }

    /// Returns true when tracing output is enabled, false otherwise.
    pub fn get_trace(&self) -> bool {
        self.trace
    }

    /// Sets tracing output.
    pub fn set_trace(&mut self, t: bool) {
        self.trace = t;
    }

    /// Returns true if the graph data to be read by read_file(sting) are in 
    /// numeric form (e.g., integer values starting from zero) or in string form.
    pub fn get_numeric(&self) -> bool {
        self.numeric
    }

    /// Specifies whether the graph data to be read by read_file(sting) 
    /// are in numeric form (e.g., integer values starting from zero) 
    /// or in string form.
    pub fn set_numeric(&mut self, n: bool) {
        self.numeric = n;
    }

    /// Returns the delimeter used in the graph data file. The data
    /// file is composed of lines with the following format:
    /// <from><delim><to>
    /// where from and to are the two graph vertices (can be either strings or
    /// integers) and delim is the delimiter.
    pub fn get_delim(&self) -> &str {
        &self.delim
    }

    /// Sets the delimited to be used for reading the graph data file.
    pub fn set_delim(&mut self, d: &str) {
        self.delim = d.to_string();
    }

    /// Outputs the parameters of the pagerank algorithm to the
    /// given output stream. The parameters are:
    /// - the damping factor (alpha)
    /// - the convergence criterion (convergence)
    /// - the maximum number of iterations (max iterations)
    /// - whether numeric or string input is expected (numeric)
    /// - the delimiter for separating the two vertices in each line of the
    ///   input file (delim)
    pub fn print_params(&self) {
        println!("alpha = {} convergence = {} max_iterations = {} numeric = {} delimiter = '{}'", 
            self.alpha, self.convergence, self.max_iterations, self.numeric, self.delim);
    }

    /// Outputs the hyperlink table.
    pub fn print_table(&self) {
        let mut i = 0;
        for cr in &self.rows {
            print!("{}:[ ", i);
            for cc in cr {
                if self.numeric {
                    print!("{} ", cc);
                } else {
                    print!("{} ", self.idx_to_nodes[cc]);
                }
            }
            print!("]\n");
            i += 1;
        }
    }

    /// Outputs the number of outgoing links for each vertex of the 
    /// hyperlink table.
    pub fn print_outgoing(&self) {
        print!("[ ");
        for cn in &self.num_outgoing {
            print!("{} ", cn);
        }
        print!("]\n");
    }

    /// Prints the pagerank vector to cout. The output format is a
    /// series of lines:
    /// <node> = <pagerank value> followed by a line:
    /// s = <sum> where <sum> is the sum of the pagerank values, which
    /// should be equal to one.
    pub fn print_pagerank(&self) {
        let mut sum: f64 = 0.0;

        print!("({}) [ ", self.pr.len());
        for cr in &self.pr {
            print!("{:10} ", cr);
            sum += *cr;
            print!("s = {} ", sum);
        }
        
        print!("] {}\n", sum);
    }

    /// Outputs the pageranks vector in a more verbose way than print_pagerank():
    /// it substitutes string vertex names for numeric IDs, if available,
    /// and also outputs the index number of each vector, starting from zero.
    pub fn print_pagerank_v(&self) {
        let mut i = 0;
        let num_rows = self.pr.len();
        let mut sum = 0.0;

        while i < num_rows {
            if !self.numeric {
                println!("{} = {}", self.idx_to_nodes[&i], self.pr[i]);
            } else {
                println!("{} = {}", i, self.pr[i]);
            }
            sum += self.pr[i];

            i += 1;
        }

        print!("s = {} \n", sum);
    }
}