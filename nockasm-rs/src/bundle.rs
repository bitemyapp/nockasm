//! Named, multi-root Nockasm DAG bundles and their compact binary envelope.

use std::collections::HashSet;
use std::fmt;

use crate::dag::{lower_nodes, LiftState, Mode};
use crate::{Atom, DagError, DagId, DagNode, DagOp, Noun};

/// Version of the compact binary DAG bundle envelope.
pub const NASM_BUNDLE_VERSION: u32 = 1;

const MAGIC: &[u8; 8] = b"NSDAGB01";

/// How a named input root is interpreted during lifting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DagMode {
    /// Interpret formula-position cells using Nock's opcode grammar.
    Formula,
    /// Preserve the noun structurally without interpreting opcodes.
    Noun,
}

impl DagMode {
    fn internal(self) -> Mode {
        match self {
            Self::Formula => Mode::Formula,
            Self::Noun => Mode::Noun,
        }
    }
}

/// One named input to [`lift_bundle`].
#[derive(Clone, Copy, Debug)]
pub struct DagInput<'a> {
    /// Stable root name stored in the bundle.
    pub name: &'a str,
    /// Noun read from this root.
    pub noun: &'a Noun,
    /// Formula or structural reading mode.
    pub mode: DagMode,
}

/// One named root in a [`NasmBundle`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DagRoot {
    name: String,
    mode: DagMode,
    id: DagId,
}

impl DagRoot {
    /// Root name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Reading mode used to construct this root.
    pub fn mode(&self) -> DagMode {
        self.mode
    }

    /// Root node ID.
    pub fn id(&self) -> DagId {
        self.id
    }
}

/// A sharing-preserving node table with one or more named roots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NasmBundle {
    nodes: Vec<DagNode>,
    roots: Vec<DagRoot>,
}

impl NasmBundle {
    /// Unique nodes in topological order.
    pub fn nodes(&self) -> &[DagNode] {
        &self.nodes
    }

    /// Named roots in input order.
    pub fn roots(&self) -> &[DagRoot] {
        &self.roots
    }

    /// Rebuild every named root while retaining sharing across roots.
    pub fn lower(&self) -> Vec<(String, Noun)> {
        let values = lower_nodes(&self.nodes);
        self.roots
            .iter()
            .map(|root| (root.name.clone(), values[root.id.index()].clone()))
            .collect()
    }

    /// Rebuild one root by name.
    pub fn lower_root(&self, name: &str) -> Option<Noun> {
        let root = self.roots.iter().find(|root| root.name == name)?;
        let values = lower_nodes(&self.nodes);
        Some(values[root.id.index()].clone())
    }

    /// Encode the bundle into the compact, versioned binary envelope.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.nodes.len().saturating_mul(8));
        out.extend_from_slice(MAGIC);
        put_varint(&mut out, u64::from(NASM_BUNDLE_VERSION));
        put_varint(&mut out, self.nodes.len() as u64);
        for node in &self.nodes {
            encode_node(&mut out, node);
        }
        put_varint(&mut out, self.roots.len() as u64);
        for root in &self.roots {
            put_varint(&mut out, root.name.len() as u64);
            out.extend_from_slice(root.name.as_bytes());
            out.push(match root.mode {
                DagMode::Formula => 0,
                DagMode::Noun => 1,
            });
            put_varint(&mut out, root.id.0.into());
        }
        out
    }

    /// Decode and validate a compact binary bundle.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BundleError> {
        let mut input = Decoder::new(bytes);
        if input.take(MAGIC.len())? != MAGIC {
            return Err(input.error("bad Nockasm bundle magic"));
        }
        let version = input.varint()?;
        if version != u64::from(NASM_BUNDLE_VERSION) {
            return Err(input.error(format!("unsupported bundle version {version}")));
        }
        let node_count = input.usize("node count")?;
        if node_count > u32::MAX as usize {
            return Err(input.error("bundle has more than u32::MAX nodes"));
        }
        if node_count > input.remaining().len() {
            return Err(input.error("node count exceeds remaining bundle bytes"));
        }
        let mut nodes = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            nodes.push(decode_node(&mut input, nodes.len())?);
        }
        let root_count = input.usize("root count")?;
        if root_count == 0 {
            return Err(input.error("bundle contains no roots"));
        }
        if root_count > input.remaining().len() {
            return Err(input.error("root count exceeds remaining bundle bytes"));
        }
        let mut names = HashSet::with_capacity(root_count);
        let mut roots = Vec::with_capacity(root_count);
        for _ in 0..root_count {
            let name_len = input.usize("root name length")?;
            let name = std::str::from_utf8(input.take(name_len)?)
                .map_err(|err| input.error(format!("root name is not UTF-8: {err}")))?
                .to_string();
            if name.is_empty() {
                return Err(input.error("root name is empty"));
            }
            if !names.insert(name.clone()) {
                return Err(input.error(format!("duplicate root name {name:?}")));
            }
            let mode = match input.byte()? {
                0 => DagMode::Formula,
                1 => DagMode::Noun,
                value => return Err(input.error(format!("invalid root mode {value}"))),
            };
            let id = input.id(nodes.len())?;
            roots.push(DagRoot { name, mode, id });
        }
        if !input.remaining().is_empty() {
            return Err(input.error("trailing bytes after bundle"));
        }
        Ok(Self { nodes, roots })
    }
}

/// Lift multiple roots into one node table, preserving sharing across roots.
pub fn lift_bundle(inputs: &[DagInput<'_>]) -> Result<NasmBundle, BundleError> {
    if inputs.is_empty() {
        return Err(BundleError::NoRoots);
    }
    let mut names = HashSet::with_capacity(inputs.len());
    let mut state = LiftState::new();
    let mut roots = Vec::with_capacity(inputs.len());
    for input in inputs {
        if input.name.is_empty() {
            return Err(BundleError::InvalidRootName("root name is empty".into()));
        }
        if !names.insert(input.name) {
            return Err(BundleError::InvalidRootName(format!(
                "duplicate root name {:?}",
                input.name
            )));
        }
        let id = state.lift(input.noun, input.mode.internal())?;
        roots.push(DagRoot {
            name: input.name.to_string(),
            mode: input.mode,
            id,
        });
    }
    Ok(NasmBundle {
        nodes: state.nodes,
        roots,
    })
}

/// Bundle construction or binary decoding failure.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BundleError {
    /// DAG lifting failed.
    Dag(DagError),
    /// The caller supplied no roots.
    NoRoots,
    /// A root name was empty or duplicated.
    InvalidRootName(String),
    /// A malformed binary bundle.
    InvalidBinary {
        /// Byte offset nearest the failure.
        offset: usize,
        /// Human-readable reason.
        message: String,
    },
}

impl From<DagError> for BundleError {
    fn from(value: DagError) -> Self {
        Self::Dag(value)
    }
}

impl fmt::Display for BundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dag(error) => error.fmt(f),
            Self::NoRoots => write!(f, "Nockasm bundle contains no roots"),
            Self::InvalidRootName(message) => write!(f, "invalid Nockasm root: {message}"),
            Self::InvalidBinary { offset, message } => {
                write!(f, "invalid Nockasm bundle at byte {offset}: {message}")
            }
        }
    }
}

impl std::error::Error for BundleError {}

fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn put_id(out: &mut Vec<u8>, id: DagId) {
    put_varint(out, id.0.into());
}

fn put_atom(out: &mut Vec<u8>, atom: &Atom) {
    let bytes = atom.to_le_bytes();
    put_varint(out, bytes.len() as u64);
    out.extend_from_slice(&bytes);
}

fn encode_node(out: &mut Vec<u8>, node: &DagNode) {
    match node {
        DagNode::Atom(atom) => {
            out.push(0);
            put_atom(out, atom);
        }
        DagNode::Cell(head, tail) => {
            out.push(1);
            put_id(out, *head);
            put_id(out, *tail);
        }
        DagNode::Nock(raw) => {
            out.push(2);
            put_id(out, *raw);
        }
        DagNode::Op(op) => encode_op(out, op),
    }
}

fn encode_op(out: &mut Vec<u8>, op: &DagOp) {
    macro_rules! ids {
        ($tag:expr, $($id:expr),+ $(,)?) => {{
            out.push($tag);
            $(put_id(out, $id);)+
        }};
    }
    match op {
        DagOp::Slot(axis) => {
            out.push(3);
            put_atom(out, axis);
        }
        DagOp::Const(value) => ids!(4, *value),
        DagOp::Eval(subject, formula) => ids!(5, *subject, *formula),
        DagOp::Isa(formula) => ids!(6, *formula),
        DagOp::Inc(formula) => ids!(7, *formula),
        DagOp::Eq(left, right) => ids!(8, *left, *right),
        DagOp::If(condition, then_, else_) => ids!(9, *condition, *then_, *else_),
        DagOp::Comp(first, second) => ids!(10, *first, *second),
        DagOp::Push(value, body) => ids!(11, *value, *body),
        DagOp::Call(axis, formula) => {
            out.push(12);
            put_atom(out, axis);
            put_id(out, *formula);
        }
        DagOp::Edit(axis, value, formula) => {
            out.push(13);
            put_atom(out, axis);
            put_id(out, *value);
            put_id(out, *formula);
        }
        DagOp::Hint(tag, formula) => ids!(14, *tag, *formula),
        DagOp::Hintd(tag, clue, formula) => ids!(15, *tag, *clue, *formula),
        DagOp::Scry(reference, path) => ids!(16, *reference, *path),
    }
}

struct Decoder<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Decoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> &'a [u8] {
        &self.bytes[self.offset..]
    }

    fn error(&self, message: impl Into<String>) -> BundleError {
        BundleError::InvalidBinary {
            offset: self.offset,
            message: message.into(),
        }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], BundleError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| self.error("offset overflow"))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| self.error("unexpected end of bundle"))?;
        self.offset = end;
        Ok(value)
    }

    fn byte(&mut self) -> Result<u8, BundleError> {
        Ok(self.take(1)?[0])
    }

    fn varint(&mut self) -> Result<u64, BundleError> {
        let mut value = 0u64;
        for shift in (0..=63).step_by(7) {
            let byte = self.byte()?;
            if shift == 63 && byte > 1 {
                return Err(self.error("varint overflow"));
            }
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(self.error("varint overflow"))
    }

    fn usize(&mut self, label: &str) -> Result<usize, BundleError> {
        usize::try_from(self.varint()?)
            .map_err(|_| self.error(format!("{label} does not fit usize")))
    }

    fn id(&mut self, bound: usize) -> Result<DagId, BundleError> {
        let raw = self.varint()?;
        let id = u32::try_from(raw).map_err(|_| self.error("node ID does not fit u32"))?;
        if id as usize >= bound {
            return Err(self.error(format!("node reference @{id} is outside 0..{bound}")));
        }
        Ok(DagId(id))
    }

    fn atom(&mut self) -> Result<Atom, BundleError> {
        let len = self.usize("atom length")?;
        Ok(Atom::from_le_bytes(self.take(len)?))
    }
}

fn decode_node(input: &mut Decoder<'_>, bound: usize) -> Result<DagNode, BundleError> {
    let id = |input: &mut Decoder<'_>| input.id(bound);
    Ok(match input.byte()? {
        0 => DagNode::Atom(input.atom()?),
        1 => DagNode::Cell(id(input)?, id(input)?),
        2 => DagNode::Nock(id(input)?),
        3 => DagNode::Op(DagOp::Slot(input.atom()?)),
        4 => DagNode::Op(DagOp::Const(id(input)?)),
        5 => DagNode::Op(DagOp::Eval(id(input)?, id(input)?)),
        6 => DagNode::Op(DagOp::Isa(id(input)?)),
        7 => DagNode::Op(DagOp::Inc(id(input)?)),
        8 => DagNode::Op(DagOp::Eq(id(input)?, id(input)?)),
        9 => DagNode::Op(DagOp::If(id(input)?, id(input)?, id(input)?)),
        10 => DagNode::Op(DagOp::Comp(id(input)?, id(input)?)),
        11 => DagNode::Op(DagOp::Push(id(input)?, id(input)?)),
        12 => DagNode::Op(DagOp::Call(input.atom()?, id(input)?)),
        13 => DagNode::Op(DagOp::Edit(input.atom()?, id(input)?, id(input)?)),
        14 => DagNode::Op(DagOp::Hint(id(input)?, id(input)?)),
        15 => DagNode::Op(DagOp::Hintd(id(input)?, id(input)?, id(input)?)),
        16 => DagNode::Op(DagOp::Scry(id(input)?, id(input)?)),
        tag => return Err(input.error(format!("unknown node tag {tag}"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{jam, noun};

    #[test]
    fn multi_root_bundle_shares_nodes_and_round_trips_binary() {
        let shared = noun![4 0 1];
        let raw = Noun::cell(shared.clone(), shared.clone());
        let bundle = lift_bundle(&[
            DagInput {
                name: "formula",
                noun: &shared,
                mode: DagMode::Formula,
            },
            DagInput {
                name: "raw",
                noun: &raw,
                mode: DagMode::Noun,
            },
        ])
        .unwrap();

        let decoded = NasmBundle::from_bytes(&bundle.to_bytes()).unwrap();
        assert_eq!(decoded, bundle);
        assert_eq!(decoded.lower_root("formula"), Some(shared));
        assert_eq!(decoded.lower_root("raw"), Some(raw));
    }

    #[test]
    fn structural_root_round_trip_matches_jam() {
        let shared = noun![100 200 300];
        let noun = Noun::cell(shared.clone(), shared);
        let bundle = lift_bundle(&[DagInput {
            name: "root",
            noun: &noun,
            mode: DagMode::Noun,
        }])
        .unwrap();
        assert_eq!(jam(&bundle.lower_root("root").unwrap()), jam(&noun));
    }

    #[test]
    fn malformed_binary_is_rejected() {
        let noun = noun![0 1];
        let bundle = lift_bundle(&[DagInput {
            name: "root",
            noun: &noun,
            mode: DagMode::Formula,
        }])
        .unwrap();
        let mut bytes = bundle.to_bytes();
        bytes.truncate(bytes.len() - 1);
        assert!(NasmBundle::from_bytes(&bytes).is_err());
        assert!(NasmBundle::from_bytes(b"wat").is_err());
    }
}
