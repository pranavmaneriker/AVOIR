Tracking for Optimization
=========================


* The optimal delta given observations must be computed by solving an optimizaiton problem. This must be initialized separately from the parsing of the specification initially to initialize all terms with distinct identifiers in the tree.

* The hierarchy of terms in the parsed DSL tree (as per the grammar) is `Specification > SpecificationThreshold > ExpectationTerm > Expectation > NumericalExpression`

* Once the initial specification is parsed, we call the `prepare_for_opt` at the top-level specification. This recursively initializes identifiers for all the terms in the tree with each identifier corresponding to the path from the specification to that node.

* Once each node in the tree has a unique identifier, we can initialize the optimization problem, recursively, passing down the same optimization problem from the root. While this problem is being passed down, each Expectation and ETerm node can add its params to the optimization problem.

* 