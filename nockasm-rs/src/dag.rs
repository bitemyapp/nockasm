//! DAG-preserving noun-to-Nockasm conversion and textual serialization.
//!
//! Ordinary [`crate::Nasm`] is an ergonomic boxed tree. Compiled kernels are
//! not trees: JAM backreferences commonly collapse millions of repeated
//! formula occurrences into a much smaller noun DAG. [`NasmDag`] retains that
//! sharing with stable node IDs, and its line-oriented text format writes each
//! unique node exactly once. Both conversion and rendering are O(unique nodes),
//! not O(the fully expanded tree).

use std::collections::HashMap;
use std::fmt;

use crate::noun::{Atom, Noun, NounRef};

/// Version of the canonical DAG text envelope.
pub const NASM_DAG_VERSION: u32 = 1;

/// A stable index into a [`NasmDag`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DagId(pub(crate) u32);

impl DagId {
    /// The zero-based node index.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// One opcode node in a DAG-lifted formula.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DagOp {
    /// `[0 axis]`.
    Slot(Atom),
    /// `[1 noun]`.
    Const(DagId),
    /// `[2 subject formula]`.
    Eval(DagId, DagId),
    /// `[3 formula]`.
    Isa(DagId),
    /// `[4 formula]`.
    Inc(DagId),
    /// `[5 left right]`.
    Eq(DagId, DagId),
    /// `[6 condition then else]`.
    If(DagId, DagId, DagId),
    /// `[7 first second]`.
    Comp(DagId, DagId),
    /// `[8 value body]`.
    Push(DagId, DagId),
    /// `[9 axis formula]`.
    Call(Atom, DagId),
    /// `[10 [axis value] formula]`.
    Edit(Atom, DagId, DagId),
    /// `[11 tag formula]`.
    Hint(DagId, DagId),
    /// `[11 [tag clue] formula]`.
    Hintd(DagId, DagId, DagId),
    /// `[12 reference path]`.
    Scry(DagId, DagId),
}

/// One unique node in a [`NasmDag`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DagNode {
    /// A raw atom in noun position.
    Atom(Atom),
    /// A binary raw cell. Binary shape retains exact sharing; the ordinary
    /// renderer may flatten the same right spine for presentation.
    Cell(DagId, DagId),
    /// A named Nock opcode.
    Op(DagOp),
    /// An opaque formula fallback whose child is read structurally.
    Nock(DagId),
}

/// A DAG-preserving Nockasm AST.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NasmDag {
    pub(crate) nodes: Vec<DagNode>,
    pub(crate) root: DagId,
}

impl NasmDag {
    /// Nodes in topological order. Every referenced ID is less than the node
    /// that references it.
    pub fn nodes(&self) -> &[DagNode] {
        &self.nodes
    }

    /// The root formula node.
    pub fn root(&self) -> DagId {
        self.root
    }

    /// Rebuild the represented noun while retaining DAG sharing.
    pub fn lower(&self) -> Noun {
        let values = lower_nodes(&self.nodes);
        values[self.root.index()].clone()
    }

    /// Render the canonical, DAG-preserving textual form.
    pub fn render(&self) -> String {
        let mut out = String::with_capacity(self.nodes.len().saturating_mul(24));
        self.write_to(&mut out)
            .expect("writing to String cannot fail");
        out
    }

    /// Stream the canonical DAG text to a formatting sink.
    pub fn write_to(&self, out: &mut impl fmt::Write) -> fmt::Result {
        writeln!(out, ":nockasm-dag {NASM_DAG_VERSION}")?;
        for (index, node) in self.nodes.iter().enumerate() {
            write!(out, "@{index} ")?;
            write_node(out, node)?;
            out.write_char('\n')?;
        }
        writeln!(out, "@root @{}", self.root.index())
    }
}

pub(crate) fn lower_nodes(nodes: &[DagNode]) -> Vec<Noun> {
    let mut values: Vec<Noun> = Vec::with_capacity(nodes.len());
    for node in nodes {
        let noun = match node {
            DagNode::Atom(atom) => Noun::from(atom.clone()),
            DagNode::Cell(head, tail) => {
                Noun::cell(values[head.index()].clone(), values[tail.index()].clone())
            }
            DagNode::Nock(raw) => values[raw.index()].clone(),
            DagNode::Op(op) => lower_op(op, &values),
        };
        values.push(noun);
    }
    values
}

/// A malformed DAG text or an input too large for 32-bit node IDs.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DagError {
    /// More than `u32::MAX` unique nodes were required.
    TooManyNodes,
    /// A malformed line in DAG text.
    InvalidText {
        /// One-based input line.
        line: usize,
        /// Human-readable reason.
        message: String,
    },
    /// No `@root` line was present.
    MissingRoot,
}

impl fmt::Display for DagError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DagError::TooManyNodes => write!(f, "nockasm DAG exceeds 32-bit node IDs"),
            DagError::InvalidText { line, message } => {
                write!(f, "invalid nockasm DAG at line {line}: {message}")
            }
            DagError::MissingRoot => write!(f, "nockasm DAG has no @root line"),
        }
    }
}

impl std::error::Error for DagError {}

fn noun_list(items: impl IntoIterator<Item = Noun>) -> Noun {
    Noun::autocons(items.into_iter().collect()).expect("opcode lists contain at least two nouns")
}

fn lower_op(op: &DagOp, values: &[Noun]) -> Noun {
    let get = |id: DagId| values[id.index()].clone();
    match op {
        DagOp::Slot(axis) => Noun::cell(0u64, Noun::from(axis.clone())),
        DagOp::Const(value) => Noun::cell(1u64, get(*value)),
        DagOp::Eval(subject, formula) => {
            noun_list([Noun::from(2u64), get(*subject), get(*formula)])
        }
        DagOp::Isa(formula) => Noun::cell(3u64, get(*formula)),
        DagOp::Inc(formula) => Noun::cell(4u64, get(*formula)),
        DagOp::Eq(left, right) => noun_list([Noun::from(5u64), get(*left), get(*right)]),
        DagOp::If(condition, then_, else_) => {
            noun_list([Noun::from(6u64), get(*condition), get(*then_), get(*else_)])
        }
        DagOp::Comp(first, second) => noun_list([Noun::from(7u64), get(*first), get(*second)]),
        DagOp::Push(value, body) => noun_list([Noun::from(8u64), get(*value), get(*body)]),
        DagOp::Call(axis, formula) => {
            noun_list([Noun::from(9u64), Noun::from(axis.clone()), get(*formula)])
        }
        DagOp::Edit(axis, value, formula) => noun_list([
            Noun::from(10u64),
            Noun::cell(Noun::from(axis.clone()), get(*value)),
            get(*formula),
        ]),
        DagOp::Hint(tag, formula) => noun_list([Noun::from(11u64), get(*tag), get(*formula)]),
        DagOp::Hintd(tag, clue, formula) => noun_list([
            Noun::from(11u64),
            Noun::cell(get(*tag), get(*clue)),
            get(*formula),
        ]),
        DagOp::Scry(reference, path) => noun_list([Noun::from(12u64), get(*reference), get(*path)]),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Formula,
    Noun,
}

enum Build {
    Cell,
    Nock,
    Const,
    Eval,
    Isa,
    Inc,
    Eq,
    If,
    Comp,
    Push,
    Call(Atom),
    Edit(Atom),
    Hint,
    Hintd,
    Scry,
}

enum Task<'a> {
    Visit(&'a Noun, Mode),
    Build {
        noun: &'a Noun,
        mode: Mode,
        kind: Build,
        children: usize,
    },
}

fn memo<'a>(
    mode: Mode,
    formula: &'a mut HashMap<Noun, DagId>,
    noun: &'a mut HashMap<Noun, DagId>,
) -> &'a mut HashMap<Noun, DagId> {
    match mode {
        Mode::Formula => formula,
        Mode::Noun => noun,
    }
}

fn push_node(nodes: &mut Vec<DagNode>, node: DagNode) -> Result<DagId, DagError> {
    let id = u32::try_from(nodes.len()).map_err(|_| DagError::TooManyNodes)?;
    nodes.push(node);
    Ok(DagId(id))
}

fn schedule<'a>(
    tasks: &mut Vec<Task<'a>>,
    noun: &'a Noun,
    mode: Mode,
    kind: Build,
    children: &[(&'a Noun, Mode)],
) {
    tasks.push(Task::Build {
        noun,
        mode,
        kind,
        children: children.len(),
    });
    for &(child, child_mode) in children.iter().rev() {
        tasks.push(Task::Visit(child, child_mode));
    }
}

fn schedule_nock<'a>(tasks: &mut Vec<Task<'a>>, noun: &'a Noun) {
    schedule(
        tasks,
        noun,
        Mode::Formula,
        Build::Nock,
        &[(noun, Mode::Noun)],
    );
}

pub(crate) struct LiftState {
    pub(crate) nodes: Vec<DagNode>,
    formula_memo: HashMap<Noun, DagId>,
    noun_memo: HashMap<Noun, DagId>,
}

impl LiftState {
    pub(crate) fn new() -> Self {
        Self {
            nodes: Vec::new(),
            formula_memo: HashMap::new(),
            noun_memo: HashMap::new(),
        }
    }

    pub(crate) fn lift(&mut self, root: &Noun, root_mode: Mode) -> Result<DagId, DagError> {
        let mut tasks = vec![Task::Visit(root, root_mode)];
        let mut values: Vec<DagId> = Vec::new();

        while let Some(task) = tasks.pop() {
            match task {
                Task::Visit(noun, mode) => {
                    if let Some(id) =
                        memo(mode, &mut self.formula_memo, &mut self.noun_memo).get(noun)
                    {
                        values.push(*id);
                        continue;
                    }
                    match mode {
                        Mode::Noun => match noun.view() {
                            NounRef::Atom(atom) => {
                                let id = push_node(&mut self.nodes, DagNode::Atom(atom.clone()))?;
                                self.noun_memo.insert(noun.clone(), id);
                                values.push(id);
                            }
                            NounRef::Cell(head, tail) => schedule(
                                &mut tasks,
                                noun,
                                mode,
                                Build::Cell,
                                &[(head, mode), (tail, mode)],
                            ),
                        },
                        Mode::Formula => visit_formula(
                            noun,
                            &mut tasks,
                            &mut self.nodes,
                            &mut self.formula_memo,
                            &mut values,
                        )?,
                    }
                }
                Task::Build {
                    noun,
                    mode,
                    kind,
                    children,
                } => {
                    let split = values.len() - children;
                    let ids = values.split_off(split);
                    let node = match (kind, ids.as_slice()) {
                        (Build::Cell, [head, tail]) => DagNode::Cell(*head, *tail),
                        (Build::Nock, [raw]) => DagNode::Nock(*raw),
                        (Build::Const, [value]) => DagNode::Op(DagOp::Const(*value)),
                        (Build::Eval, [subject, formula]) => {
                            DagNode::Op(DagOp::Eval(*subject, *formula))
                        }
                        (Build::Isa, [formula]) => DagNode::Op(DagOp::Isa(*formula)),
                        (Build::Inc, [formula]) => DagNode::Op(DagOp::Inc(*formula)),
                        (Build::Eq, [left, right]) => DagNode::Op(DagOp::Eq(*left, *right)),
                        (Build::If, [condition, then_, else_]) => {
                            DagNode::Op(DagOp::If(*condition, *then_, *else_))
                        }
                        (Build::Comp, [first, second]) => DagNode::Op(DagOp::Comp(*first, *second)),
                        (Build::Push, [value, body]) => DagNode::Op(DagOp::Push(*value, *body)),
                        (Build::Call(axis), [formula]) => DagNode::Op(DagOp::Call(axis, *formula)),
                        (Build::Edit(axis), [value, formula]) => {
                            DagNode::Op(DagOp::Edit(axis, *value, *formula))
                        }
                        (Build::Hint, [tag, formula]) => DagNode::Op(DagOp::Hint(*tag, *formula)),
                        (Build::Hintd, [tag, clue, formula]) => {
                            DagNode::Op(DagOp::Hintd(*tag, *clue, *formula))
                        }
                        (Build::Scry, [reference, path]) => {
                            DagNode::Op(DagOp::Scry(*reference, *path))
                        }
                        _ => unreachable!("build arity is fixed by scheduling"),
                    };
                    let id = push_node(&mut self.nodes, node)?;
                    memo(mode, &mut self.formula_memo, &mut self.noun_memo)
                        .insert(noun.clone(), id);
                    values.push(id);
                }
            }
        }

        debug_assert_eq!(values.len(), 1);
        Ok(values.pop().expect("root produces one DAG node"))
    }
}

/// Lift a noun as a formula while preserving its DAG sharing.
///
/// Formula-position and noun-position reads have separate memo tables: the
/// same noun can correctly become an opcode in one position and raw data in
/// another. Soundness is exact: `lift_dag(n)?.lower() == n`.
pub fn lift_dag(root: &Noun) -> Result<NasmDag, DagError> {
    let mut state = LiftState::new();
    let root = state.lift(root, Mode::Formula)?;
    Ok(NasmDag {
        nodes: state.nodes,
        root,
    })
}

/// Lift an arbitrary noun structurally while preserving its DAG sharing.
///
/// Unlike [`lift_dag`], this does not assert that the root is a formula and
/// never interprets noun cells as named opcodes. It is the correct entry point
/// for compiled kernels, types, cache records, and other noun-shaped data.
pub fn lift_noun_dag(root: &Noun) -> Result<NasmDag, DagError> {
    let mut state = LiftState::new();
    let root = state.lift(root, Mode::Noun)?;
    Ok(NasmDag {
        nodes: state.nodes,
        root,
    })
}

fn visit_formula<'a>(
    noun: &'a Noun,
    tasks: &mut Vec<Task<'a>>,
    nodes: &mut Vec<DagNode>,
    formula_memo: &mut HashMap<Noun, DagId>,
    values: &mut Vec<DagId>,
) -> Result<(), DagError> {
    let (head, tail) = match noun.view() {
        NounRef::Atom(_) => {
            schedule_nock(tasks, noun);
            return Ok(());
        }
        NounRef::Cell(head, tail) => (head, tail),
    };
    let NounRef::Atom(opcode) = head.view() else {
        schedule(
            tasks,
            noun,
            Mode::Formula,
            Build::Cell,
            &[(head, Mode::Formula), (tail, Mode::Formula)],
        );
        return Ok(());
    };

    match opcode.as_u64() {
        Some(0) => match tail.view() {
            NounRef::Atom(axis) => {
                let id = push_node(nodes, DagNode::Op(DagOp::Slot(axis.clone())))?;
                formula_memo.insert(noun.clone(), id);
                values.push(id);
            }
            NounRef::Cell(..) => schedule_nock(tasks, noun),
        },
        Some(1) => schedule(
            tasks,
            noun,
            Mode::Formula,
            Build::Const,
            &[(tail, Mode::Noun)],
        ),
        Some(2) => match tail.as_cell() {
            Some((subject, formula)) if subject.is_cell() && formula.is_cell() => schedule(
                tasks,
                noun,
                Mode::Formula,
                Build::Eval,
                &[(subject, Mode::Formula), (formula, Mode::Formula)],
            ),
            _ => schedule_nock(tasks, noun),
        },
        Some(3) if tail.is_cell() => schedule(
            tasks,
            noun,
            Mode::Formula,
            Build::Isa,
            &[(tail, Mode::Formula)],
        ),
        Some(4) if tail.is_cell() => schedule(
            tasks,
            noun,
            Mode::Formula,
            Build::Inc,
            &[(tail, Mode::Formula)],
        ),
        Some(5) => match tail.as_cell() {
            Some((left, right)) if left.is_cell() && right.is_cell() => schedule(
                tasks,
                noun,
                Mode::Formula,
                Build::Eq,
                &[(left, Mode::Formula), (right, Mode::Formula)],
            ),
            _ => schedule_nock(tasks, noun),
        },
        Some(6) => match tail.as_cell() {
            Some((condition, branches)) if condition.is_cell() => match branches.as_cell() {
                Some((then_, else_)) if then_.is_cell() && else_.is_cell() => schedule(
                    tasks,
                    noun,
                    Mode::Formula,
                    Build::If,
                    &[
                        (condition, Mode::Formula),
                        (then_, Mode::Formula),
                        (else_, Mode::Formula),
                    ],
                ),
                _ => schedule_nock(tasks, noun),
            },
            _ => schedule_nock(tasks, noun),
        },
        Some(7) => schedule_binary_formula(tasks, noun, tail, Build::Comp),
        Some(8) => schedule_binary_formula(tasks, noun, tail, Build::Push),
        Some(9) => match tail.as_cell() {
            Some((axis, formula)) if formula.is_cell() => match axis.view() {
                NounRef::Atom(axis) => schedule(
                    tasks,
                    noun,
                    Mode::Formula,
                    Build::Call(axis.clone()),
                    &[(formula, Mode::Formula)],
                ),
                NounRef::Cell(..) => schedule_nock(tasks, noun),
            },
            _ => schedule_nock(tasks, noun),
        },
        Some(10) => match tail.as_cell() {
            Some((spec, formula)) if formula.is_cell() => match spec.as_cell() {
                Some((axis, value)) if value.is_cell() => match axis.view() {
                    NounRef::Atom(axis) => schedule(
                        tasks,
                        noun,
                        Mode::Formula,
                        Build::Edit(axis.clone()),
                        &[(value, Mode::Formula), (formula, Mode::Formula)],
                    ),
                    NounRef::Cell(..) => schedule_nock(tasks, noun),
                },
                _ => schedule_nock(tasks, noun),
            },
            _ => schedule_nock(tasks, noun),
        },
        Some(11) => match tail.as_cell() {
            Some((tag, formula)) if tag.is_atom() && formula.is_cell() => schedule(
                tasks,
                noun,
                Mode::Formula,
                Build::Hint,
                &[(tag, Mode::Noun), (formula, Mode::Formula)],
            ),
            Some((spec, formula)) if formula.is_cell() => match spec.as_cell() {
                Some((tag, clue)) if clue.is_cell() => schedule(
                    tasks,
                    noun,
                    Mode::Formula,
                    Build::Hintd,
                    &[
                        (tag, Mode::Noun),
                        (clue, Mode::Formula),
                        (formula, Mode::Formula),
                    ],
                ),
                _ => schedule_nock(tasks, noun),
            },
            _ => schedule_nock(tasks, noun),
        },
        Some(12) => schedule_binary_formula(tasks, noun, tail, Build::Scry),
        _ => schedule_nock(tasks, noun),
    }
    Ok(())
}

fn schedule_binary_formula<'a>(
    tasks: &mut Vec<Task<'a>>,
    noun: &'a Noun,
    tail: &'a Noun,
    kind: Build,
) {
    match tail.as_cell() {
        Some((left, right)) if left.is_cell() && right.is_cell() => schedule(
            tasks,
            noun,
            Mode::Formula,
            kind,
            &[(left, Mode::Formula), (right, Mode::Formula)],
        ),
        _ => schedule_nock(tasks, noun),
    }
}

fn write_id(out: &mut impl fmt::Write, id: DagId) -> fmt::Result {
    write!(out, "@{}", id.index())
}

fn write_atom(out: &mut impl fmt::Write, atom: &Atom) -> fmt::Result {
    if let Some(value) = atom.as_u64() {
        return write!(out, "{value}");
    }
    out.write_str("0x")?;
    for byte in atom.le_bytes().iter().rev() {
        write!(out, "{byte:02x}")?;
    }
    Ok(())
}

fn write_node(out: &mut impl fmt::Write, node: &DagNode) -> fmt::Result {
    match node {
        DagNode::Atom(atom) => {
            out.write_str("atom ")?;
            write_atom(out, atom)
        }
        DagNode::Cell(head, tail) => {
            out.write_str("cell ")?;
            write_id(out, *head)?;
            out.write_char(' ')?;
            write_id(out, *tail)
        }
        DagNode::Nock(raw) => {
            out.write_str("nock ")?;
            write_id(out, *raw)
        }
        DagNode::Op(op) => write_op(out, op),
    }
}

fn write_op(out: &mut impl fmt::Write, op: &DagOp) -> fmt::Result {
    macro_rules! ids {
        ($name:literal, $($id:expr),+ $(,)?) => {{
            out.write_str($name)?;
            $(out.write_char(' ')?; write_id(out, $id)?;)+
            Ok(())
        }};
    }
    match op {
        DagOp::Slot(axis) => {
            out.write_str("slot ")?;
            write_atom(out, axis)
        }
        DagOp::Const(value) => ids!("const", *value),
        DagOp::Eval(subject, formula) => ids!("eval", *subject, *formula),
        DagOp::Isa(formula) => ids!("isa", *formula),
        DagOp::Inc(formula) => ids!("inc", *formula),
        DagOp::Eq(left, right) => ids!("eq", *left, *right),
        DagOp::If(condition, then_, else_) => ids!("if", *condition, *then_, *else_),
        DagOp::Comp(first, second) => ids!("comp", *first, *second),
        DagOp::Push(value, body) => ids!("push", *value, *body),
        DagOp::Call(axis, formula) => {
            out.write_str("call ")?;
            write_atom(out, axis)?;
            out.write_char(' ')?;
            write_id(out, *formula)
        }
        DagOp::Edit(axis, value, formula) => {
            out.write_str("edit ")?;
            write_atom(out, axis)?;
            out.write_char(' ')?;
            write_id(out, *value)?;
            out.write_char(' ')?;
            write_id(out, *formula)
        }
        DagOp::Hint(tag, formula) => ids!("hint", *tag, *formula),
        DagOp::Hintd(tag, clue, formula) => ids!("hintd", *tag, *clue, *formula),
        DagOp::Scry(reference, path) => ids!("scry", *reference, *path),
    }
}

fn text_error(line: usize, message: impl Into<String>) -> DagError {
    DagError::InvalidText {
        line,
        message: message.into(),
    }
}

fn parse_id(token: &str, line: usize, bound: usize) -> Result<DagId, DagError> {
    let raw = token
        .strip_prefix('@')
        .ok_or_else(|| text_error(line, format!("expected @id, got {token:?}")))?;
    let id = raw
        .parse::<u32>()
        .map_err(|err| text_error(line, format!("bad node ID {token:?}: {err}")))?;
    if id as usize >= bound {
        return Err(text_error(
            line,
            format!("reference {token} is not an earlier node"),
        ));
    }
    Ok(DagId(id))
}

fn parse_atom(token: &str, line: usize) -> Result<Atom, DagError> {
    if let Some(hex) = token.strip_prefix("0x") {
        if hex.is_empty() || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(text_error(line, format!("invalid hex atom {token:?}")));
        }
        return Ok(Atom::from_hex_digits(hex));
    }
    if token.is_empty() || !token.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(text_error(line, format!("invalid decimal atom {token:?}")));
    }
    Ok(Atom::from_decimal_digits(token))
}

fn expect_arity<'a>(
    fields: &'a [&'a str],
    want: usize,
    line: usize,
) -> Result<&'a [&'a str], DagError> {
    if fields.len() != want {
        return Err(text_error(
            line,
            format!(
                "{} takes {} fields, got {}",
                fields[0],
                want - 1,
                fields.len() - 1
            ),
        ));
    }
    Ok(fields)
}

fn parse_node(fields: &[&str], line: usize, bound: usize) -> Result<DagNode, DagError> {
    let op = *fields
        .first()
        .ok_or_else(|| text_error(line, "missing node kind"))?;
    let id = |token: &str| parse_id(token, line, bound);
    let atom = |token: &str| parse_atom(token, line);
    Ok(match op {
        "atom" => {
            let f = expect_arity(fields, 2, line)?;
            DagNode::Atom(atom(f[1])?)
        }
        "cell" => {
            let f = expect_arity(fields, 3, line)?;
            DagNode::Cell(id(f[1])?, id(f[2])?)
        }
        "nock" => {
            let f = expect_arity(fields, 2, line)?;
            DagNode::Nock(id(f[1])?)
        }
        "slot" => {
            let f = expect_arity(fields, 2, line)?;
            DagNode::Op(DagOp::Slot(atom(f[1])?))
        }
        "const" => unary(fields, line, bound, DagOp::Const)?,
        "eval" => binary(fields, line, bound, DagOp::Eval)?,
        "isa" => unary(fields, line, bound, DagOp::Isa)?,
        "inc" => unary(fields, line, bound, DagOp::Inc)?,
        "eq" => binary(fields, line, bound, DagOp::Eq)?,
        "if" => ternary(fields, line, bound, DagOp::If)?,
        "comp" => binary(fields, line, bound, DagOp::Comp)?,
        "push" => binary(fields, line, bound, DagOp::Push)?,
        "call" => {
            let f = expect_arity(fields, 3, line)?;
            DagNode::Op(DagOp::Call(atom(f[1])?, id(f[2])?))
        }
        "edit" => {
            let f = expect_arity(fields, 4, line)?;
            DagNode::Op(DagOp::Edit(atom(f[1])?, id(f[2])?, id(f[3])?))
        }
        "hint" => binary(fields, line, bound, DagOp::Hint)?,
        "hintd" => ternary(fields, line, bound, DagOp::Hintd)?,
        "scry" => binary(fields, line, bound, DagOp::Scry)?,
        _ => return Err(text_error(line, format!("unknown node kind {op:?}"))),
    })
}

fn unary(
    fields: &[&str],
    line: usize,
    bound: usize,
    build: fn(DagId) -> DagOp,
) -> Result<DagNode, DagError> {
    let f = expect_arity(fields, 2, line)?;
    Ok(DagNode::Op(build(parse_id(f[1], line, bound)?)))
}

fn binary(
    fields: &[&str],
    line: usize,
    bound: usize,
    build: fn(DagId, DagId) -> DagOp,
) -> Result<DagNode, DagError> {
    let f = expect_arity(fields, 3, line)?;
    Ok(DagNode::Op(build(
        parse_id(f[1], line, bound)?,
        parse_id(f[2], line, bound)?,
    )))
}

fn ternary(
    fields: &[&str],
    line: usize,
    bound: usize,
    build: fn(DagId, DagId, DagId) -> DagOp,
) -> Result<DagNode, DagError> {
    let f = expect_arity(fields, 4, line)?;
    Ok(DagNode::Op(build(
        parse_id(f[1], line, bound)?,
        parse_id(f[2], line, bound)?,
        parse_id(f[3], line, bound)?,
    )))
}

/// Parse canonical DAG text emitted by [`NasmDag::render`].
pub fn parse_dag(source: &str) -> Result<NasmDag, DagError> {
    let mut lines = source.lines().enumerate();
    let Some((_, header)) = lines.next() else {
        return Err(text_error(1, "missing :nockasm-dag header"));
    };
    if header != format!(":nockasm-dag {NASM_DAG_VERSION}") {
        return Err(text_error(1, format!("unsupported header {header:?}")));
    }

    let mut nodes = Vec::new();
    let mut root = None;
    for (zero_line, raw) in lines {
        let line = zero_line + 1;
        let text = raw.trim();
        if text.is_empty() {
            continue;
        }
        let fields = text.split_whitespace().collect::<Vec<_>>();
        if fields.first() == Some(&"@root") {
            let f = expect_arity(&fields, 2, line)?;
            if root.is_some() {
                return Err(text_error(line, "duplicate @root"));
            }
            root = Some(parse_id(f[1], line, nodes.len())?);
            continue;
        }
        if root.is_some() {
            return Err(text_error(line, "node appears after @root"));
        }
        let expected = format!("@{}", nodes.len());
        if fields.first().copied() != Some(expected.as_str()) {
            return Err(text_error(
                line,
                format!("expected node {expected}, got {:?}", fields.first()),
            ));
        }
        nodes.push(parse_node(&fields[1..], line, nodes.len())?);
    }
    let root = root.ok_or(DagError::MissingRoot)?;
    Ok(NasmDag { nodes, root })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noun;

    #[test]
    fn shared_formula_is_not_expanded() {
        let shared = noun![4 0 1];
        let formula = Noun::cell(shared.clone(), shared);
        let dag = lift_dag(&formula).unwrap();
        assert_eq!(dag.lower(), formula);
        assert!(dag.nodes().len() < 10, "sharing should keep the DAG small");
    }

    #[test]
    fn text_round_trip_preserves_sharing_and_scry() {
        let shared = noun![12 [1 138] [0 1]];
        let formula = Noun::cell(shared.clone(), shared);
        let dag = lift_dag(&formula).unwrap();
        let text = dag.render();
        let reparsed = parse_dag(&text).unwrap();
        assert_eq!(reparsed, dag);
        assert_eq!(reparsed.lower(), formula);
        assert_eq!(reparsed.render(), text);
    }

    #[test]
    fn malformed_text_is_rejected() {
        assert!(parse_dag(":nockasm-dag 1\n@0 cell @0 @0\n@root @0\n").is_err());
        assert!(parse_dag(":nockasm-dag 9\n").is_err());
        assert!(parse_dag(":nockasm-dag 1\n@0 atom wat\n@root @0\n").is_err());
    }
}
