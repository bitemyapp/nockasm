//! `noted`: the annotated instantiation of the IR vocabulary.
//!
//! [`Nasm`] is the IR the rest of this crate consumes, and it is a
//! closed tree: every child is another `Nasm`. A compiler emitting
//! nockasm usually wants to carry something alongside each node — a
//! source position for diagnostics, a provenance tag — and the
//! tempting move is to fork the vocabulary: a second enum with the same
//! variants plus an annotation, drifting from the first every time a
//! variant is added.
//!
//! The Hoon reference (`desk/sur/nockasm.hoon`) avoids the fork by
//! parameterizing the vocabulary over its own recursion, so that an
//! annotated instantiation is a first-class citizen rather than a copy:
//!
//! ```text
//! ++  nasm-of  |$  [self]  $%  [%atom p=@]  ...  [%let p=@t q=self r=self]  ...  ==
//! +$  nasm   $~([%atom 0] (nasm-of nasm))                        ::  the plain knot
//! +$  noted  $~([*note [%atom 0]] [=note node=(nasm-of noted)])  ::  an annotated one
//! ```
//!
//! This module is that pattern in Rust. [`NasmOf<N>`] (with [`OpOf<N>`]
//! and [`MatchArmOf<N>`]) is the vocabulary generic over the child
//! type, mirroring [`Nasm`], [`Op`], and [`MatchArm`] variant for
//! variant; [`Noted<A>`] ties the knot through a wrapper that carries a
//! note of type `A` on every node. The intended consumer is a compiler
//! that emits positioned IR — the Jock backend instantiates the Hoon
//! builder with `[pos=(unit hair) node=(nasm-of nasm)]`, which here is
//! `Noted<Option<Pos>>`.
//!
//! # The strip contract
//!
//! [`Noted::strip`] is the projection onto the bare IR: drop every
//! note, recurse. [`Noted::from_nasm`] is a section of it — every node
//! gets the same note — so `Noted::from_nasm(&n, a).strip() == n` for
//! every `n`. The meaning of an annotated value is the meaning of its
//! projection, `lower(schema, &noted.strip())`; the conformance law a
//! positioned emitter is held to is `(expand ours) == (reference
//! expansion of (strip ours))`. [`lower`] and [`render`] here are
//! exactly the bare pipeline composed with the projection, so a
//! compiler holding annotated IR never has to spell it out.
//!
//! ```
//! use nockasm::noted::{self, Noted};
//! use nockasm::{expand, parse};
//!
//! let src = ":subject .x  #let .d = (%inc .x) in .d";
//! let program = parse(src).unwrap();
//! // A compiler builds `Noted` directly, positions and all; here every
//! // node gets the same note.
//! let positioned: Noted<Option<(u32, u32)>> =
//!     Noted::from_nasm(&program.body, Some((1, 1)));
//! assert_eq!(positioned.strip(), program.body);
//! assert_eq!(
//!     noted::lower(program.schema.as_ref(), &positioned).unwrap(),
//!     expand(src).unwrap(),
//! );
//! ```
//!
//! # Drift is a compile error
//!
//! `Nasm` stays a separate concrete type — it is the cross-implementation
//! contract and what every other stage consumes — so the bare and generic
//! vocabularies are kept in step by hand, and this module makes a slip
//! fail to compile rather than fail at runtime. The bridge between them
//! is exactly two functions: [`unroll`] (`&Nasm -> NasmOf<&Nasm>`) is an
//! exhaustive match over `Nasm` and `Op`, and [`roll`]
//! (`NasmOf<Nasm> -> Nasm`) is an exhaustive match over `NasmOf` and
//! `OpOf`. Neither has a wildcard arm, so a variant added to either side
//! without its twin breaks one of them — the same discipline the Hoon
//! `$opco` term set imposes on its expander. Everything else in the
//! module ([`strip`](Noted::strip), [`from_nasm`](Noted::from_nasm),
//! [`map_note`](Noted::map_note), the teardown) is built from those two
//! and from the vocabulary's own [`map`](NasmOf::map), so it inherits a
//! new variant automatically.
//!
//! # Depth
//!
//! `Noted` is a boxed tree like `Nasm` and keeps the same guarantee:
//! teardown, `strip`, `from_nasm`, and `map_note` all run on explicit
//! stacks, so annotated IR of any depth converts and drops without
//! touching the call stack. As with `Nasm`, the *derived* `Clone`,
//! `PartialEq`, and `Debug` recurse; deep-cloning or deep-comparing
//! annotated IR is the caller's lookout.

use crate::ast::{MatchArm, Name, Nasm, Op, Schema};
use crate::error::LowerError;
use crate::noun::{Atom, Noun};

// ----------------------------------------------------------------------
// The vocabulary, generic over its recursion
// ----------------------------------------------------------------------

/// One layer of the IR vocabulary with every child position replaced
/// by `N`: [`Nasm`] variant for variant, minus the recursion.
///
/// `NasmOf<&Nasm>` is a borrowed view of one plain node (see
/// [`unroll`]), `NasmOf<Nasm>` an owned one (see [`roll`]), and
/// `NasmOf<Box<Noted<A>>>` the node of an annotated tree. The
/// per-variant semantics are those of the [`Nasm`] variant of the same
/// name; only the child type differs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NasmOf<N> {
    /// An atom literal. See [`Nasm::Atom`].
    Atom(Atom),
    /// `.name` — a reference into the subject schema. See
    /// [`Nasm::Axis`].
    Axis(Name),
    /// `[a b c ...]` — a raw structural cell of two-or-more elements.
    /// See [`Nasm::Cell`].
    Cell {
        /// First element.
        first: N,
        /// Second element.
        second: N,
        /// Any further elements.
        rest: Vec<N>,
    },
    /// `(%opcode ...)` — a named opcode application. See [`Nasm::Op`].
    Op(OpOf<N>),
    /// `#let name = value in body`. See [`Nasm::Let`].
    Let {
        /// The bound name (axis 2 in the body).
        name: Name,
        /// The pushed value — a formula position.
        value: N,
        /// The body — a formula position.
        body: N,
    },
    /// `#match scrutinee { pat => body ... _ => default }`. See
    /// [`Nasm::Match`].
    Match {
        /// The scrutinee — a formula position.
        scrutinee: N,
        /// The literal-pattern arms, in source order.
        arms: Vec<MatchArmOf<N>>,
        /// The required `_ =>` default — a formula position.
        default: N,
    },
    /// `(%nock F)` — an already-formed formula embedded as an opaque
    /// noun. See [`Nasm::Nock`].
    Nock(Noun),
}

/// A well-formed named-opcode application with child positions of type
/// `N`: [`Op`] variant for variant. Argument kinds (formula, noun, axis)
/// are those of the [`Op`] variant of the same name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpOf<N> {
    /// `(%slot N)` → `[0 N]`. See [`Op::Slot`].
    Slot(Atom),
    /// `(%self)` → `[0 1]`. See [`Op::Self_`].
    Self_,
    /// `(%battery)` → `[0 2]`. See [`Op::Battery`].
    Battery,
    /// `(%payload)` → `[0 3]`. See [`Op::Payload`].
    Payload,
    /// `(%sample)` → `[0 6]`. See [`Op::Sample`].
    Sample,
    /// `(%context)` → `[0 7]`. See [`Op::Context`].
    Context,
    /// `(%crash)` → `[0 0]`. See [`Op::Crash`].
    Crash,
    /// `(%const X)` → `[1 X]`. See [`Op::Const`].
    Const(N),
    /// `(%arm X)` → `[1 X]`. See [`Op::Arm`].
    Arm(N),
    /// `(%eval S F)` → `[2 S F]`. See [`Op::Eval`].
    Eval(N, N),
    /// `(%isa F)` → `[3 F]`. See [`Op::Isa`].
    Isa(N),
    /// `(%inc F)` → `[4 F]`. See [`Op::Inc`].
    Inc(N),
    /// `(%eq F G)` → `[5 F G]`. See [`Op::Eq`].
    Eq(N, N),
    /// `(%if C T E)` → `[6 C T E]`. See [`Op::If`].
    If(N, N, N),
    /// `(%comp F G)` → `[7 F G]`. See [`Op::Comp`].
    Comp(N, N),
    /// `(%push F G)` → `[8 F G]`. See [`Op::Push`].
    Push(N, N),
    /// `(%call N F)` → `[9 N F]`. See [`Op::Call`].
    Call(Atom, N),
    /// `(%edit N V F)` → `[10 [N V] F]`. See [`Op::Edit`].
    Edit(Atom, N, N),
    /// `(%hint T F)` → `[11 T F]`. See [`Op::Hint`].
    Hint(N, N),
    /// `(%hintd T C F)` → `[11 [T C] F]`. See [`Op::Hintd`].
    Hintd(N, N, N),
    /// `(%scry R P)` → `[12 R P]`. See [`Op::Scry`].
    Scry(N, N),
}

/// One `#match` arm with positions of type `N`: [`MatchArm`] minus the
/// recursion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatchArmOf<N> {
    /// The literal pattern — a noun position.
    pub pattern: N,
    /// The arm body — a formula position.
    pub body: N,
}

impl<N> NasmOf<N> {
    /// Replace every child, in source order, keeping the node's own
    /// data (atoms, names, the `%nock` payload).
    ///
    /// This is the vocabulary's functorial map, and the one place the
    /// child order is defined: first, second, then the rest of a cell;
    /// an opcode's arguments left to right; a `#let`'s value then body;
    /// a `#match`'s scrutinee, then each arm's pattern and body, then
    /// the default. Every traversal in this module enumerates children
    /// through it. Exhaustive over `NasmOf`, deliberately: see the
    /// module docs.
    pub fn map<M>(self, mut f: impl FnMut(N) -> M) -> NasmOf<M> {
        match self {
            NasmOf::Atom(a) => NasmOf::Atom(a),
            NasmOf::Axis(name) => NasmOf::Axis(name),
            NasmOf::Cell {
                first,
                second,
                rest,
            } => {
                let first = f(first);
                let second = f(second);
                let rest = rest.into_iter().map(&mut f).collect();
                NasmOf::Cell {
                    first,
                    second,
                    rest,
                }
            }
            NasmOf::Op(op) => NasmOf::Op(op.map(f)),
            NasmOf::Let { name, value, body } => {
                let value = f(value);
                let body = f(body);
                NasmOf::Let { name, value, body }
            }
            NasmOf::Match {
                scrutinee,
                arms,
                default,
            } => {
                let scrutinee = f(scrutinee);
                let arms = arms.into_iter().map(|arm| arm.map(&mut f)).collect();
                let default = f(default);
                NasmOf::Match {
                    scrutinee,
                    arms,
                    default,
                }
            }
            NasmOf::Nock(noun) => NasmOf::Nock(noun),
        }
    }

    /// Borrow every child, cloning the node's own data (atoms and
    /// nouns are refcounted; a name is one string clone).
    /// `node.as_ref().map(f)` is the by-reference map.
    pub fn as_ref(&self) -> NasmOf<&N> {
        match self {
            NasmOf::Atom(a) => NasmOf::Atom(a.clone()),
            NasmOf::Axis(name) => NasmOf::Axis(name.clone()),
            NasmOf::Cell {
                first,
                second,
                rest,
            } => NasmOf::Cell {
                first,
                second,
                rest: rest.iter().collect(),
            },
            NasmOf::Op(op) => NasmOf::Op(op.as_ref()),
            NasmOf::Let { name, value, body } => NasmOf::Let {
                name: name.clone(),
                value,
                body,
            },
            NasmOf::Match {
                scrutinee,
                arms,
                default,
            } => NasmOf::Match {
                scrutinee,
                arms: arms.iter().map(MatchArmOf::as_ref).collect(),
                default,
            },
            NasmOf::Nock(noun) => NasmOf::Nock(noun.clone()),
        }
    }

    /// No child positions at all. (A zero-argument opcode is not a
    /// leaf by this definition, matching `Nasm`; it merely has nothing
    /// to push.) `Nock` holds a `Noun`, whose own drop is iterative.
    fn is_leaf(&self) -> bool {
        matches!(self, NasmOf::Atom(_) | NasmOf::Axis(_) | NasmOf::Nock(_))
    }
}

impl<N> OpOf<N> {
    /// The opcode's source name, without the `%`; agrees with
    /// [`Op::name`].
    pub fn name(&self) -> &'static str {
        match self {
            OpOf::Slot(_) => "slot",
            OpOf::Self_ => "self",
            OpOf::Battery => "battery",
            OpOf::Payload => "payload",
            OpOf::Sample => "sample",
            OpOf::Context => "context",
            OpOf::Crash => "crash",
            OpOf::Const(_) => "const",
            OpOf::Arm(_) => "arm",
            OpOf::Eval(..) => "eval",
            OpOf::Isa(_) => "isa",
            OpOf::Inc(_) => "inc",
            OpOf::Eq(..) => "eq",
            OpOf::If(..) => "if",
            OpOf::Comp(..) => "comp",
            OpOf::Push(..) => "push",
            OpOf::Call(..) => "call",
            OpOf::Edit(..) => "edit",
            OpOf::Hint(..) => "hint",
            OpOf::Hintd(..) => "hintd",
            OpOf::Scry(..) => "scry",
        }
    }

    /// Replace every argument, left to right, keeping axis atoms.
    /// Exhaustive over `OpOf`, deliberately: see the module docs.
    pub fn map<M>(self, mut f: impl FnMut(N) -> M) -> OpOf<M> {
        match self {
            OpOf::Slot(ax) => OpOf::Slot(ax),
            OpOf::Self_ => OpOf::Self_,
            OpOf::Battery => OpOf::Battery,
            OpOf::Payload => OpOf::Payload,
            OpOf::Sample => OpOf::Sample,
            OpOf::Context => OpOf::Context,
            OpOf::Crash => OpOf::Crash,
            OpOf::Const(x) => OpOf::Const(f(x)),
            OpOf::Arm(x) => OpOf::Arm(f(x)),
            OpOf::Eval(a, b) => {
                let a = f(a);
                OpOf::Eval(a, f(b))
            }
            OpOf::Isa(x) => OpOf::Isa(f(x)),
            OpOf::Inc(x) => OpOf::Inc(f(x)),
            OpOf::Eq(a, b) => {
                let a = f(a);
                OpOf::Eq(a, f(b))
            }
            OpOf::If(c, t, e) => {
                let c = f(c);
                let t = f(t);
                OpOf::If(c, t, f(e))
            }
            OpOf::Comp(a, b) => {
                let a = f(a);
                OpOf::Comp(a, f(b))
            }
            OpOf::Push(a, b) => {
                let a = f(a);
                OpOf::Push(a, f(b))
            }
            OpOf::Call(ax, x) => OpOf::Call(ax, f(x)),
            OpOf::Edit(ax, v, x) => {
                let v = f(v);
                OpOf::Edit(ax, v, f(x))
            }
            OpOf::Hint(t, x) => {
                let t = f(t);
                OpOf::Hint(t, f(x))
            }
            OpOf::Hintd(t, c, x) => {
                let t = f(t);
                let c = f(c);
                OpOf::Hintd(t, c, f(x))
            }
            OpOf::Scry(r, p) => {
                let r = f(r);
                OpOf::Scry(r, f(p))
            }
        }
    }

    /// Borrow every argument, cloning axis atoms.
    pub fn as_ref(&self) -> OpOf<&N> {
        match self {
            OpOf::Slot(ax) => OpOf::Slot(ax.clone()),
            OpOf::Self_ => OpOf::Self_,
            OpOf::Battery => OpOf::Battery,
            OpOf::Payload => OpOf::Payload,
            OpOf::Sample => OpOf::Sample,
            OpOf::Context => OpOf::Context,
            OpOf::Crash => OpOf::Crash,
            OpOf::Const(x) => OpOf::Const(x),
            OpOf::Arm(x) => OpOf::Arm(x),
            OpOf::Eval(a, b) => OpOf::Eval(a, b),
            OpOf::Isa(x) => OpOf::Isa(x),
            OpOf::Inc(x) => OpOf::Inc(x),
            OpOf::Eq(a, b) => OpOf::Eq(a, b),
            OpOf::If(c, t, e) => OpOf::If(c, t, e),
            OpOf::Comp(a, b) => OpOf::Comp(a, b),
            OpOf::Push(a, b) => OpOf::Push(a, b),
            OpOf::Call(ax, x) => OpOf::Call(ax.clone(), x),
            OpOf::Edit(ax, v, x) => OpOf::Edit(ax.clone(), v, x),
            OpOf::Hint(t, x) => OpOf::Hint(t, x),
            OpOf::Hintd(t, c, x) => OpOf::Hintd(t, c, x),
            OpOf::Scry(r, p) => OpOf::Scry(r, p),
        }
    }
}

impl<N> MatchArmOf<N> {
    /// Replace the pattern, then the body.
    pub fn map<M>(self, mut f: impl FnMut(N) -> M) -> MatchArmOf<M> {
        let pattern = f(self.pattern);
        let body = f(self.body);
        MatchArmOf { pattern, body }
    }

    /// Borrow both positions.
    pub fn as_ref(&self) -> MatchArmOf<&N> {
        MatchArmOf {
            pattern: &self.pattern,
            body: &self.body,
        }
    }
}

// ----------------------------------------------------------------------
// The bridge to the plain instantiation
// ----------------------------------------------------------------------

/// View one layer of plain IR through the vocabulary, children
/// borrowed: the `Nasm ≅ NasmOf<Nasm>` isomorphism, read one way.
///
/// Exhaustive over [`Nasm`] and [`Op`] with no wildcard arm — this is
/// where a variant added to `Nasm` without its `NasmOf` twin fails to
/// compile (see the module docs). Node data is cloned: atoms and nouns
/// are refcounted, a name is one string clone.
pub fn unroll(n: &Nasm) -> NasmOf<&Nasm> {
    match n {
        Nasm::Atom(a) => NasmOf::Atom(a.clone()),
        Nasm::Axis(name) => NasmOf::Axis(name.clone()),
        Nasm::Cell {
            first,
            second,
            rest,
        } => NasmOf::Cell {
            first,
            second,
            rest: rest.iter().collect(),
        },
        Nasm::Op(op) => NasmOf::Op(match op {
            Op::Slot(ax) => OpOf::Slot(ax.clone()),
            Op::Self_ => OpOf::Self_,
            Op::Battery => OpOf::Battery,
            Op::Payload => OpOf::Payload,
            Op::Sample => OpOf::Sample,
            Op::Context => OpOf::Context,
            Op::Crash => OpOf::Crash,
            Op::Const(x) => OpOf::Const(x),
            Op::Arm(x) => OpOf::Arm(x),
            Op::Eval(a, b) => OpOf::Eval(a, b),
            Op::Isa(x) => OpOf::Isa(x),
            Op::Inc(x) => OpOf::Inc(x),
            Op::Eq(a, b) => OpOf::Eq(a, b),
            Op::If(c, t, e) => OpOf::If(c, t, e),
            Op::Comp(a, b) => OpOf::Comp(a, b),
            Op::Push(a, b) => OpOf::Push(a, b),
            Op::Call(ax, x) => OpOf::Call(ax.clone(), x),
            Op::Edit(ax, v, x) => OpOf::Edit(ax.clone(), v, x),
            Op::Hint(t, x) => OpOf::Hint(t, x),
            Op::Hintd(t, c, x) => OpOf::Hintd(t, c, x),
            Op::Scry(r, p) => OpOf::Scry(r, p),
        }),
        Nasm::Let { name, value, body } => NasmOf::Let {
            name: name.clone(),
            value,
            body,
        },
        Nasm::Match {
            scrutinee,
            arms,
            default,
        } => NasmOf::Match {
            scrutinee,
            arms: arms
                .iter()
                .map(|arm| MatchArmOf {
                    pattern: &arm.pattern,
                    body: &arm.body,
                })
                .collect(),
            default,
        },
        Nasm::Nock(noun) => NasmOf::Nock(noun.clone()),
    }
}

/// Close one layer of the vocabulary over plain children: the
/// `Nasm ≅ NasmOf<Nasm>` isomorphism, read the other way.
///
/// Exhaustive over [`NasmOf`] and [`OpOf`] with no wildcard arm — this
/// is where a variant added to `NasmOf` without its `Nasm` twin fails
/// to compile (see the module docs).
pub fn roll(layer: NasmOf<Nasm>) -> Nasm {
    match layer {
        NasmOf::Atom(a) => Nasm::Atom(a),
        NasmOf::Axis(name) => Nasm::Axis(name),
        NasmOf::Cell {
            first,
            second,
            rest,
        } => Nasm::Cell {
            first: Box::new(first),
            second: Box::new(second),
            rest,
        },
        NasmOf::Op(op) => Nasm::Op(match op {
            OpOf::Slot(ax) => Op::Slot(ax),
            OpOf::Self_ => Op::Self_,
            OpOf::Battery => Op::Battery,
            OpOf::Payload => Op::Payload,
            OpOf::Sample => Op::Sample,
            OpOf::Context => Op::Context,
            OpOf::Crash => Op::Crash,
            OpOf::Const(x) => Op::Const(Box::new(x)),
            OpOf::Arm(x) => Op::Arm(Box::new(x)),
            OpOf::Eval(a, b) => Op::Eval(Box::new(a), Box::new(b)),
            OpOf::Isa(x) => Op::Isa(Box::new(x)),
            OpOf::Inc(x) => Op::Inc(Box::new(x)),
            OpOf::Eq(a, b) => Op::Eq(Box::new(a), Box::new(b)),
            OpOf::If(c, t, e) => Op::If(Box::new(c), Box::new(t), Box::new(e)),
            OpOf::Comp(a, b) => Op::Comp(Box::new(a), Box::new(b)),
            OpOf::Push(a, b) => Op::Push(Box::new(a), Box::new(b)),
            OpOf::Call(ax, x) => Op::Call(ax, Box::new(x)),
            OpOf::Edit(ax, v, x) => Op::Edit(ax, Box::new(v), Box::new(x)),
            OpOf::Hint(t, x) => Op::Hint(Box::new(t), Box::new(x)),
            OpOf::Hintd(t, c, x) => Op::Hintd(Box::new(t), Box::new(c), Box::new(x)),
            OpOf::Scry(r, p) => Op::Scry(Box::new(r), Box::new(p)),
        }),
        NasmOf::Let { name, value, body } => Nasm::Let {
            name,
            value: Box::new(value),
            body: Box::new(body),
        },
        NasmOf::Match {
            scrutinee,
            arms,
            default,
        } => Nasm::Match {
            scrutinee: Box::new(scrutinee),
            arms: arms
                .into_iter()
                .map(|arm| MatchArm {
                    pattern: arm.pattern,
                    body: arm.body,
                })
                .collect(),
            default: Box::new(default),
        },
        NasmOf::Nock(noun) => Nasm::Nock(noun),
    }
}

// ----------------------------------------------------------------------
// The post-order driver
// ----------------------------------------------------------------------

/// One pending step of a [`fold`].
enum Task<'a, S> {
    /// Open this node and schedule its children.
    Visit(&'a S),
    /// Every child of `layer` has a result on the value stack; refill
    /// the layer with them and close it.
    Build {
        source: &'a S,
        layer: NasmOf<&'a S>,
        count: usize,
    },
}

/// Post-order fold over any tree that reads as layers of the
/// vocabulary, on an explicit stack: `open` views one node with its
/// children borrowed, `close` builds a node's result from the node and
/// the results of its children. Children are enumerated through
/// [`NasmOf::map`] itself, so the order they are visited in and the
/// order their results are handed back in cannot disagree.
fn fold<'a, S, R>(
    root: &'a S,
    open: impl Fn(&'a S) -> NasmOf<&'a S>,
    mut close: impl FnMut(&'a S, NasmOf<R>) -> R,
) -> R {
    let mut tasks: Vec<Task<'a, S>> = vec![Task::Visit(root)];
    let mut done: Vec<R> = Vec::new();
    while let Some(task) = tasks.pop() {
        match task {
            Task::Visit(source) => {
                let mut kids: Vec<&'a S> = Vec::new();
                let layer = open(source).map(|child| {
                    kids.push(child);
                    child
                });
                tasks.push(Task::Build {
                    source,
                    layer,
                    count: kids.len(),
                });
                // Leftmost child on top, so it completes first and its
                // result sits deepest in `done`.
                tasks.extend(kids.into_iter().rev().map(Task::Visit));
            }
            Task::Build {
                source,
                layer,
                count,
            } => {
                let mut results = done
                    .drain(done.len() - count..)
                    .collect::<Vec<R>>()
                    .into_iter();
                let filled = layer.map(|_| results.next().expect("one result per child"));
                debug_assert!(results.next().is_none(), "no result left over");
                done.push(close(source, filled));
            }
        }
    }
    debug_assert_eq!(done.len(), 1, "every fold yields one result");
    done.pop().expect("the root result")
}

// ----------------------------------------------------------------------
// The annotated instantiation
// ----------------------------------------------------------------------

/// The annotated instantiation of the vocabulary: every node carries a
/// note of type `A` beside its [`NasmOf`] layer, whose children are
/// again `Noted<A>`.
///
/// `Noted<()>` is plain IR in all but name; `Noted<Option<Pos>>` is a
/// positioned IR. The fields are public — a compiler builds these
/// directly — but, like [`Nasm`], the type implements `Drop` (so that
/// deep trees tear down iteratively), which means it cannot be
/// destructured by value; borrow the fields, or `strip` / `map_note`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Noted<A> {
    /// The annotation.
    pub note: A,
    /// The node, with annotated children.
    pub node: NasmOf<Box<Noted<A>>>,
}

impl<A> Noted<A> {
    /// One layer with the children borrowed through their boxes.
    fn open(&self) -> NasmOf<&Noted<A>> {
        self.node.as_ref().map(|child| &**child)
    }

    /// Project onto the bare IR: drop every note, recurse.
    ///
    /// This is the strip contract of the module docs —
    /// `Noted::from_nasm(&n, a).strip() == n` — and the meaning of an
    /// annotated value is the meaning of its projection. Runs on an
    /// explicit stack; total; every node's data is cloned once (atoms
    /// and nouns are refcounted, names are string clones).
    pub fn strip(&self) -> Nasm {
        fold(self, Noted::open, |_, layer| roll(layer))
    }

    /// Rewrite every note, keeping the tree: `strip` of the result is
    /// `strip` of the input. Notes are visited in post-order (children
    /// before parents, left to right), on an explicit stack.
    ///
    /// By reference rather than by value: `Noted` implements `Drop`, so
    /// moving a note out of it is not expressible in safe Rust, and
    /// this crate forbids unsafe. A `FnMut(&A) -> B` covers the
    /// by-value shape anyway (clone inside the closure when `B = A`).
    pub fn map_note<B>(&self, mut f: impl FnMut(&A) -> B) -> Noted<B> {
        fold(self, Noted::open, |n, layer| Noted {
            note: f(&n.note),
            node: layer.map(Box::new),
        })
    }
}

impl<A: Clone> Noted<A> {
    /// Annotate plain IR with the same note on every node: a section
    /// of [`strip`](Noted::strip), so `Noted::from_nasm(&n, a).strip()
    /// == n` for every `n`. Runs on an explicit stack; total.
    pub fn from_nasm(n: &Nasm, note: A) -> Noted<A> {
        fold(n, unroll, |_, layer| Noted {
            note: note.clone(),
            node: layer.map(Box::new),
        })
    }
}

/// Move the children of a non-leaf node onto `stack`, leaving a leaf in
/// its place. The children's own nodes are untouched here — each is
/// hollowed in turn when it is popped — so no node is ever dropped
/// with more than one live layer beneath it.
fn hollow<A>(node: &mut NasmOf<Box<Noted<A>>>, stack: &mut Vec<Box<Noted<A>>>) {
    if node.is_leaf() {
        return;
    }
    std::mem::replace(node, NasmOf::Atom(Atom::ZERO)).map(|child| stack.push(child));
}

impl<A> Drop for Noted<A> {
    /// Iterative teardown, in the style of `Nasm`'s: a positioned
    /// emitter can produce IR as deep as its input, so dropping must
    /// not recurse. Each popped child is hollowed into `stack` before
    /// it drops, leaving only its note and its box for the normal glue
    /// to free — and its own `Drop`, finding a leaf, returns at once.
    fn drop(&mut self) {
        if self.node.is_leaf() {
            return;
        }
        let mut stack: Vec<Box<Noted<A>>> = Vec::new();
        hollow(&mut self.node, &mut stack);
        while let Some(mut child) = stack.pop() {
            hollow(&mut child.node, &mut stack);
        }
    }
}

// ----------------------------------------------------------------------
// The pipeline, composed with the projection
// ----------------------------------------------------------------------

/// Lower annotated IR to its canonical Nock noun:
/// [`crate::lower`]`(schema, &expr.strip())`.
pub fn lower<A>(schema: Option<&Schema>, expr: &Noted<A>) -> Result<Noun, LowerError> {
    crate::lower(schema, &expr.strip())
}

/// Render annotated IR to canonical `.nasm` source:
/// [`crate::render`]`(schema, &expr.strip())`. Notes do not render —
/// the canonical text is byte-identical across implementations and
/// carries no annotations by design.
pub fn render<A>(schema: Option<&Schema>, expr: &Noted<A>) -> String {
    crate::render(schema, &expr.strip())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::{expand, noun, parse, Program};

    const TEST_STACK: usize = 2 * 1024 * 1024;

    /// Every vocabulary case and every opcode, plus the mixed forms,
    /// as source (the parser is the most convenient constructor).
    const SAMPLES: &[&str] = &[
        "42",
        "0x1.0000",
        "'fast'",
        "123.456.789.012.345.678.901.234.567.890",
        ":subject {.a .b}  .b",
        "[4 0 1]",
        "[8 [1 0] 4 0 6]",
        "[1 2 3 4]",
        "(%slot 7)",
        "(%self)",
        "(%battery)",
        "(%payload)",
        "(%sample)",
        "(%context)",
        "(%crash)",
        "(%const [1 2])",
        "(%arm (%if (%slot 1) 0 1))",
        "(%eval (%slot 1) (%const 42))",
        "(%isa (%slot 1))",
        "(%inc (%slot 1))",
        "(%eq (%slot 2) (%slot 3))",
        "(%if (%slot 1) 0 1)",
        "(%comp (%slot 1) (%inc (%slot 1)))",
        "(%push (%const 42) (%slot 1))",
        "(%call 2 (%slot 1))",
        "(%edit 6 (%inc (%slot 1)) (%slot 1))",
        "(%hint 'fast' (%slot 1))",
        "(%hintd 'fast' 0 (%slot 1))",
        "(%scry (%const 138) (%slot 1))",
        "(%nock [11 'fast' 0 1])",
        "(%nock 42)",
        ":subject .x  #let .a = (%inc .x) in #let .b = (%inc .a) in (%eq .a .b)",
        ":subject {.tag .data}  #match .tag { 1 => (%inc .data)  2 => 20  _ => 0 }",
        ":subject {.before .target .after}\n#let .next = (%inc .target) in\n  [.before .next .after]\n",
    ];

    fn samples() -> Vec<(&'static str, Program)> {
        SAMPLES
            .iter()
            .map(|src| (*src, parse(src).unwrap_or_else(|e| panic!("{src}: {e}"))))
            .collect()
    }

    /// The kinds and opcode names present in a tree, by an iterative
    /// walk through `unroll`.
    fn inventory(n: &Nasm) -> (BTreeSet<&'static str>, BTreeSet<&'static str>) {
        let mut kinds = BTreeSet::new();
        let mut ops = BTreeSet::new();
        let mut stack = vec![n];
        while let Some(n) = stack.pop() {
            let layer = unroll(n);
            kinds.insert(match &layer {
                NasmOf::Atom(_) => "atom",
                NasmOf::Axis(_) => "axis",
                NasmOf::Cell { .. } => "cell",
                NasmOf::Op(op) => {
                    ops.insert(op.name());
                    "op"
                }
                NasmOf::Let { .. } => "let",
                NasmOf::Match { .. } => "match",
                NasmOf::Nock(_) => "nock",
            });
            layer.map(|child| stack.push(child));
        }
        (kinds, ops)
    }

    const ALL_OPS: &[&str] = &[
        "slot", "self", "battery", "payload", "sample", "context", "crash", "const", "arm", "eval",
        "isa", "inc", "eq", "if", "comp", "push", "call", "edit", "hint", "hintd", "scry",
    ];

    #[test]
    fn samples_cover_the_vocabulary() {
        let mut kinds = BTreeSet::new();
        let mut ops = BTreeSet::new();
        for (_, program) in samples() {
            let (k, o) = inventory(&program.body);
            kinds.extend(k);
            ops.extend(o);
        }
        let want_kinds: BTreeSet<&str> = ["atom", "axis", "cell", "op", "let", "match", "nock"]
            .into_iter()
            .collect();
        assert_eq!(kinds, want_kinds, "every Nasm variant is sampled");
        assert_eq!(
            ops,
            ALL_OPS.iter().copied().collect(),
            "every Op variant is sampled"
        );
    }

    #[test]
    fn strip_undoes_from_nasm() {
        for (src, program) in samples() {
            let unit = Noted::from_nasm(&program.body, ());
            assert_eq!(unit.strip(), program.body, "{src}");
            let tagged = Noted::from_nasm(&program.body, src);
            assert_eq!(tagged.strip(), program.body, "{src}");
        }
    }

    #[test]
    fn roll_undoes_unroll() {
        for (src, program) in samples() {
            let n = &program.body;
            assert_eq!(roll(unroll(n).map(Nasm::clone)), *n, "{src}");
        }
    }

    #[test]
    fn lower_and_render_agree_with_the_bare_pipeline() {
        for (src, program) in samples() {
            let schema = program.schema.as_ref();
            let noted = Noted::from_nasm(&program.body, 7u8);
            let want = expand(src).unwrap_or_else(|e| panic!("{src}: {e}"));
            assert_eq!(program.lower().unwrap(), want, "{src}");
            assert_eq!(lower(schema, &noted).unwrap(), want, "{src}: lower");
            assert_eq!(render(schema, &noted), program.render(), "{src}: render");
        }
    }

    #[test]
    fn from_nasm_puts_the_note_on_every_node() {
        for (src, program) in samples() {
            let noted = Noted::from_nasm(&program.body, src);
            let mut stack = vec![&noted];
            let mut count = 0usize;
            while let Some(n) = stack.pop() {
                assert_eq!(n.note, src);
                count += 1;
                n.node.as_ref().map(|child| stack.push(child));
            }
            let mut plain = 0usize;
            let mut stack = vec![&program.body];
            while let Some(n) = stack.pop() {
                plain += 1;
                unroll(n).map(|child| stack.push(child));
            }
            assert_eq!(count, plain, "{src}: one note per node");
        }
    }

    // A hand-built annotated tree, one distinct note per node, covering
    // every vocabulary case and every opcode with asymmetric arguments
    // kept distinct — so a field mislabelled in `roll`/`unroll` (a
    // swapped `%if` branch, a `#let` value for its body) shows up
    // against the parser and the expander, not just against itself.

    type Node = Box<Noted<u32>>;

    fn bx(note: u32, node: NasmOf<Node>) -> Node {
        Box::new(Noted { note, node })
    }
    fn atom(note: u32, v: u64) -> Node {
        bx(note, NasmOf::Atom(Atom::from(v)))
    }
    fn cord(note: u32, s: &str) -> Node {
        bx(note, NasmOf::Atom(Atom::from_cord(s)))
    }
    fn axis(note: u32, name: &str) -> Node {
        bx(note, NasmOf::Axis(Name::new(name).unwrap()))
    }
    fn op(note: u32, op: OpOf<Node>) -> Node {
        bx(note, NasmOf::Op(op))
    }

    const HAND_BUILT_SRC: &str = "\
:subject {.a .b}
#let .v = (%eval (%const [1 2]) (%if .a (%inc .b) (%crash))) in
#match (%edit 6 (%call 2 .v) (%hintd 'memo' (%comp .a .b) (%push (%scry .a .b) (%hint 'fast' [.a .b 7])))) {
  1 => (%eq .a .b)
  'two' => (%arm (%isa (%self)))
  3 => [(%battery) (%payload) (%sample) (%context) (%slot 5)]
  _ => (%nock [0 1])
}
";

    fn hand_built() -> Noted<u32> {
        let value = op(
            1,
            OpOf::Eval(
                op(
                    2,
                    OpOf::Const(bx(
                        3,
                        NasmOf::Cell {
                            first: atom(4, 1),
                            second: atom(5, 2),
                            rest: vec![],
                        },
                    )),
                ),
                op(
                    6,
                    OpOf::If(
                        axis(7, "a"),
                        op(8, OpOf::Inc(axis(9, "b"))),
                        op(10, OpOf::Crash),
                    ),
                ),
            ),
        );
        let scrutinee = op(
            11,
            OpOf::Edit(
                Atom::from(6u64),
                op(12, OpOf::Call(Atom::from(2u64), axis(13, "v"))),
                op(
                    14,
                    OpOf::Hintd(
                        cord(15, "memo"),
                        op(16, OpOf::Comp(axis(17, "a"), axis(18, "b"))),
                        op(
                            19,
                            OpOf::Push(
                                op(20, OpOf::Scry(axis(21, "a"), axis(22, "b"))),
                                op(
                                    23,
                                    OpOf::Hint(
                                        cord(24, "fast"),
                                        bx(
                                            25,
                                            NasmOf::Cell {
                                                first: axis(26, "a"),
                                                second: axis(27, "b"),
                                                rest: vec![atom(28, 7)],
                                            },
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );
        let arms = vec![
            MatchArmOf {
                pattern: atom(29, 1),
                body: op(30, OpOf::Eq(axis(31, "a"), axis(32, "b"))),
            },
            MatchArmOf {
                pattern: cord(33, "two"),
                body: op(34, OpOf::Arm(op(35, OpOf::Isa(op(36, OpOf::Self_))))),
            },
            MatchArmOf {
                pattern: atom(37, 3),
                body: bx(
                    38,
                    NasmOf::Cell {
                        first: op(39, OpOf::Battery),
                        second: op(40, OpOf::Payload),
                        rest: vec![
                            op(41, OpOf::Sample),
                            op(42, OpOf::Context),
                            op(43, OpOf::Slot(Atom::from(5u64))),
                        ],
                    },
                ),
            },
        ];
        let default = bx(44, NasmOf::Nock(noun![0 1]));
        let body = bx(
            45,
            NasmOf::Match {
                scrutinee,
                arms,
                default,
            },
        );
        Noted {
            note: 0,
            node: NasmOf::Let {
                name: Name::new("v").unwrap(),
                value,
                body,
            },
        }
    }

    #[test]
    fn hand_built_tree_covers_the_vocabulary() {
        let (kinds, ops) = inventory(&hand_built().strip());
        assert_eq!(kinds.len(), 7, "every Nasm variant: {kinds:?}");
        assert_eq!(ops, ALL_OPS.iter().copied().collect::<BTreeSet<_>>());
    }

    #[test]
    fn hand_built_tree_matches_parser_expander_and_renderer() {
        let program = parse(HAND_BUILT_SRC).expect("parses");
        let ours = hand_built();
        assert_eq!(ours.strip(), program.body, "strip agrees with parse");
        let schema = program.schema.as_ref();
        let want = expand(HAND_BUILT_SRC).expect("expands");
        assert_eq!(lower(schema, &ours).expect("lowers"), want);
        assert_eq!(render(schema, &ours), program.render());
        assert_eq!(
            expand(&render(schema, &ours)).expect("re-expands"),
            want,
            "round trip through the renderer"
        );
    }

    #[test]
    fn map_note_visits_every_note_once_and_keeps_the_tree() {
        let ours = hand_built();
        let mut seen: Vec<u32> = Vec::new();
        let shifted = ours.map_note(|n| {
            seen.push(*n);
            n + 100
        });
        seen.sort_unstable();
        assert_eq!(seen, (0..=45).collect::<Vec<u32>>(), "each note once");
        let mut notes: Vec<u32> = Vec::new();
        let mut stack = vec![&shifted];
        while let Some(n) = stack.pop() {
            notes.push(n.note);
            n.node.as_ref().map(|child| stack.push(child));
        }
        notes.sort_unstable();
        assert_eq!(notes, (100..=145).collect::<Vec<u32>>());
        assert_eq!(shifted.strip(), ours.strip());
        // Identity on the notes is identity on the value.
        assert_eq!(ours.map_note(|n| *n), ours);
    }

    /// The Hoon reference's `test-vocabulary-instantiation`, verbatim
    /// in shape: a span-noted tree projects onto plain IR and lowers to
    /// the expansion of the equivalent source.
    #[test]
    fn hoon_vocabulary_instantiation_test() {
        type Span = (u32, u32);
        let ex: Noted<Span> = Noted {
            note: (1, 1),
            node: NasmOf::Let {
                name: Name::new("d").unwrap(),
                value: Box::new(Noted {
                    note: (1, 10),
                    node: NasmOf::Op(OpOf::Inc(Box::new(Noted {
                        note: (1, 15),
                        node: NasmOf::Axis(Name::new("x").unwrap()),
                    }))),
                }),
                body: Box::new(Noted {
                    note: (2, 1),
                    node: NasmOf::Nock(noun![0 2]),
                }),
            },
        };
        let schema = Schema::Leaf(Name::new("x").unwrap());
        assert_eq!(
            lower(Some(&schema), &ex).expect("lowers"),
            expand(":subject .x  #let .d = (%inc .x) in (%nock [0 2])").expect("expands"),
        );
    }

    #[test]
    fn op_names_agree() {
        for (src, program) in samples() {
            let mut stack = vec![&program.body];
            while let Some(n) = stack.pop() {
                let layer = unroll(n);
                if let (Nasm::Op(op), NasmOf::Op(generic)) = (n, &layer) {
                    assert_eq!(op.name(), generic.name(), "{src}");
                }
                layer.map(|child| stack.push(child));
            }
        }
    }

    /// Conversion and teardown at a depth no recursive walk survives on
    /// a 2 MiB stack — the same discipline `Nasm` is held to.
    #[test]
    fn deep_annotated_ir_converts_and_drops_iteratively() {
        const DEPTH: usize = 100_000;
        let handle = std::thread::Builder::new()
            .stack_size(TEST_STACK)
            .spawn(|| {
                // A 100_000-deep `%inc` chain, built iteratively.
                let mut chain = Nasm::Op(Op::Slot(Atom::from(1u64)));
                for _ in 0..DEPTH {
                    chain = Nasm::Op(Op::Inc(Box::new(chain)));
                }
                let want = crate::lower(None, &chain).expect("lowers");

                // Annotate, project, renumber, and lower at depth: all
                // explicit stacks. The IR's derived `PartialEq` recurses
                // (documented), so agreement is checked through nouns,
                // whose equality is iterative.
                let annotated = Noted::from_nasm(&chain, 0u32);
                let stripped = annotated.strip();
                assert_eq!(crate::lower(None, &stripped).expect("lowers"), want);
                assert_eq!(lower(None, &annotated).expect("lowers"), want);
                let renumbered = annotated.map_note(|n| n + 1);
                assert_eq!(lower(None, &renumbered).expect("lowers"), want);
                drop(stripped);
                drop(renumbered);
                drop(annotated);
                drop(chain);

                // A chain built directly as `Noted`, so the teardown is
                // exercised independently of the conversions.
                let mut direct = Noted {
                    note: 0usize,
                    node: NasmOf::Op(OpOf::Self_),
                };
                for i in 1..=DEPTH {
                    direct = Noted {
                        note: i,
                        node: NasmOf::Op(OpOf::Inc(Box::new(direct))),
                    };
                }
                drop(direct);

                // Depth through the other child positions too: cell
                // heads and `#let` bodies.
                let mut cells = Noted {
                    note: (),
                    node: NasmOf::Atom(Atom::ZERO),
                };
                let mut lets = Noted {
                    note: (),
                    node: NasmOf::Atom(Atom::ZERO),
                };
                for _ in 0..DEPTH {
                    cells = Noted {
                        note: (),
                        node: NasmOf::Cell {
                            first: Box::new(cells),
                            second: Box::new(Noted {
                                note: (),
                                node: NasmOf::Atom(Atom::ZERO),
                            }),
                            rest: vec![],
                        },
                    };
                    lets = Noted {
                        note: (),
                        node: NasmOf::Let {
                            name: Name::new("x").unwrap(),
                            value: Box::new(Noted {
                                note: (),
                                node: NasmOf::Atom(Atom::ZERO),
                            }),
                            body: Box::new(lets),
                        },
                    };
                }
                let plain = cells.strip();
                drop(cells);
                drop(plain);
                let plain = lets.strip();
                drop(lets);
                drop(plain);
            })
            .expect("thread spawns");
        handle
            .join()
            .expect("no stack overflow on deep annotated IR");
    }
}
