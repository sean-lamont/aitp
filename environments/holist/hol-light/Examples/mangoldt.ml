(* ========================================================================= *)
(* Mangoldt function and elementary Chebyshev/Mertens results.               *)
(* ========================================================================= *)

set_jrh_lexer;;
open System;;
open Lib;;
open Fusion;;
open Basics;;
open Nets;;
open Printer;;
open Preterm;;
open Parser;;
open Equal;;
open Bool;;
open Drule;;
open Log;;
open Import_proofs;;
open Tactics;;
open Itab;;
open Replay;;
open Simp;;
open Embryo_extra;;
open Theorems;;
open Ind_defs;;
open Class;;
open Trivia;;
open Canon;;
open Meson;;
open Metis;;
open Quot;;
open Impconv;;
open Pair;;
open Nums;;
open Recursion;;
open Arith;;
open Wf;;
open Calc_num;;
open Normalizer;;
open Grobner;;
open Ind_types;;
open Lists;;
open Realax;;
open Calc_int;;
open Realarith;;
open Reals;;
open Calc_rat;;
open Ints;;
open Sets;;
open Iterate;;
open Cart;;
open Define;;
open Help;;
open Wo;;
open Binary;;
open Card;;
open Permutations;;
open Products;;
open Floor;;
open Misc;;
open Iter;;
open Vectors;;
open Determinants;;
open Topology;;
open Convex;;
open Paths;;
open Polytope;;
open Degree;;
open Derivatives;;
open Clifford;;
open Integration;;
open Measure;;

open Binomial;;
open Complexes;;
open Canal;;
open Transcendentals;;
open Realanalysis;;
open Moretop;;
open Cauchy;;

open Prime;;
open Pocklington;;

open Transcendentals;;



prioritize_real();;

(* ------------------------------------------------------------------------- *)
(* Useful approximation/bound lemmas, simple rather than sharp.              *)
(* ------------------------------------------------------------------------- *)

let LOG_FACT = prove
 (`!n. log(&(FACT n)) = sum(1..n) (\d. log(&d))`,
  INDUCT_TAC THEN
  SIMP_TAC[FACT; SUM_CLAUSES_NUMSEG; LOG_1; ARITH; ARITH_RULE `1 <= SUC n`] THEN
  SIMP_TAC[GSYM REAL_OF_NUM_MUL; LOG_MUL; REAL_OF_NUM_LT; FACT_LT; LT_0] THEN
  ASM_REWRITE_TAC[ADD1] THEN REWRITE_TAC[ADD_AC; REAL_ADD_AC]);;

let SUM_DIVISORS_FLOOR_LEMMA = prove
 (`!n d. ~(d = 0)
         ==> sum(1..n) (\m. if d divides m then &1 else &0) = floor(&n / &d)`,
  REPEAT STRIP_TAC THEN ASM_SIMP_TAC[FLOOR_DIV_DIV] THEN
  SIMP_TAC[GSYM SUM_RESTRICT_SET; FINITE_NUMSEG; SUM_CONST; FINITE_RESTRICT;
           REAL_MUL_RID; REAL_OF_NUM_EQ] THEN
  GEN_REWRITE_TAC "Examples/mangoldt.ml:RAND_CONV" RAND_CONV [GSYM CARD_NUMSEG_1] THEN
  MATCH_MP_TAC BIJECTIONS_CARD_EQ THEN
  MAP_EVERY EXISTS_TAC [`\m:num. m DIV d`; `\m:num. m * d`] THEN
  ASM_SIMP_TAC[IN_ELIM_THM; IN_NUMSEG; LE_1; DIV_MULT; DIVIDES_DIV_MULT;
               FINITE_NUMSEG; ONCE_REWRITE_RULE[MULT_SYM] DIV_MULT;
               DIV_MONO; LE_1] THEN
  ASM_SIMP_TAC[LE_RDIV_EQ; MULT_EQ_0; ARITH_RULE `1 <= n <=> ~(n = 0)`] THEN
  CONJ_TAC THENL [GEN_TAC THEN STRIP_TAC; ARITH_TAC] THEN
  FIRST_X_ASSUM(SUBST_ALL_TAC o SYM) THEN ASM_SIMP_TAC[DIV_EQ_0] THEN
  GEN_REWRITE_TAC "Examples/mangoldt.ml:(RAND_CONV o RAND_CONV)" (RAND_CONV o RAND_CONV) [ARITH_RULE `d = 1 * d`] THEN
  ASM_SIMP_TAC[LT_MULT_RCANCEL; ARITH_RULE `n < 1 <=> n = 0`] THEN
  ASM_MESON_TAC[MULT_CLAUSES]);;

let LOG_2_BOUNDS = prove
 (`&1 / &2 <= log(&2) /\ log(&2) <= &1`,
  CONJ_TAC THENL
   [GEN_REWRITE_TAC "Examples/mangoldt.ml:LAND_CONV" LAND_CONV [GSYM LOG_EXP] THEN
    MP_TAC(SPEC `inv(&2)` REAL_EXP_BOUND_LEMMA);
    GEN_REWRITE_TAC "Examples/mangoldt.ml:RAND_CONV" RAND_CONV [GSYM LOG_EXP] THEN
    MP_TAC(SPEC `&1` REAL_EXP_LE_X)] THEN
  CONV_TAC "Examples/mangoldt.ml:REAL_RAT_REDUCE_CONV" REAL_RAT_REDUCE_CONV THEN MATCH_MP_TAC EQ_IMP THEN
  CONV_TAC "Examples/mangoldt.ml:SYM_CONV" SYM_CONV THEN MATCH_MP_TAC LOG_MONO_LE THEN
  REWRITE_TAC[REAL_EXP_POS_LT; REAL_OF_NUM_LT; ARITH]);;

let LOG_LE_REFL = prove
 (`!n. ~(n = 0) ==> log(&n) <= &n`,
  REPEAT STRIP_TAC THEN
  MATCH_MP_TAC(REAL_ARITH `x <= y - &1 ==> x <= y`) THEN
  GEN_REWRITE_TAC "Examples/mangoldt.ml:(LAND_CONV o RAND_CONV)" (LAND_CONV o RAND_CONV)
   [REAL_ARITH `n = &1 + (n  - &1)`] THEN
  MATCH_MP_TAC LOG_LE THEN
  REWRITE_TAC[REAL_LE_SUB_LADD; REAL_OF_NUM_ADD; REAL_OF_NUM_LE] THEN
  UNDISCH_TAC `~(n = 0)` THEN ARITH_TAC);;

let LOG_FACT_BOUNDS = prove
 (`!n. ~(n = 0)
       ==> abs(log(&(FACT n)) - (&n * log(&n) - &n + &1)) <= &2 * log(&n)`,
  REPEAT STRIP_TAC THEN ASM_CASES_TAC `n = 1` THENL
   [ASM_REWRITE_TAC[num_CONV `1`; FACT] THEN
    REWRITE_TAC[ARITH; LOG_1] THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  ASM_SIMP_TAC[LOG_FACT] THEN
  REWRITE_TAC[REAL_ARITH `abs(x - y) <= e <=> x <= y + e /\ y - e <= x`] THEN
  CONJ_TAC THENL
   [MP_TAC(ISPECL[`\z. clog(z)`; `\z. z * clog z - z`; `1`; `n:num`]
                 SUM_INTEGRAL_UBOUND_INCREASING) THEN
    REWRITE_TAC[] THEN ANTS_TAC THENL
     [CONJ_TAC THENL [ASM_ARITH_TAC; ALL_TAC] THEN
      CONJ_TAC THENL
       [REWRITE_TAC[IN_SEGMENT_CX_GEN] THEN REPEAT STRIP_TAC THENL
         [COMPLEX_DIFF_TAC THEN CONJ_TAC THEN UNDISCH_TAC `&1 <= Re x` THENL
           [REAL_ARITH_TAC; ALL_TAC] THEN
          ASM_CASES_TAC `x = Cx(&0)` THEN ASM_REWRITE_TAC[RE_CX] THENL
           [REAL_ARITH_TAC;
            UNDISCH_TAC `~(x = Cx(&0))` THEN CONV_TAC "Examples/mangoldt.ml:COMPLEX_FIELD" COMPLEX_FIELD];
          FIRST_X_ASSUM(MP_TAC o GEN_REWRITE_RULE I [GSYM LT_NZ]) THEN
          REWRITE_TAC[GSYM REAL_OF_NUM_LT] THEN
          ASM_REAL_ARITH_TAC];
        MAP_EVERY X_GEN_TAC [`a:real`; `b:real`] THEN STRIP_TAC THEN
        SUBGOAL_THEN `&0 < a /\ &0 < b` ASSUME_TAC THENL
         [ASM_REAL_ARITH_TAC; ALL_TAC] THEN
        ASM_SIMP_TAC[GSYM CX_LOG; RE_CX; LOG_MONO_LE_IMP]];
        ALL_TAC];
     ASM_SIMP_TAC[SUM_CLAUSES_LEFT; ARITH_RULE `1 <= n <=> ~(n = 0)`] THEN
     REWRITE_TAC[LOG_1; REAL_ADD_LID; ARITH] THEN
     FIRST_ASSUM(DISJ_CASES_TAC o MATCH_MP (ARITH_RULE
      `~(n = 0) ==> n = 1 \/ 2 <= n`))
     THENL
      [ASM_REWRITE_TAC[] THEN CONV_TAC "Examples/mangoldt.ml:(ONCE_DEPTH_CONV NUMSEG_CONV)" (ONCE_DEPTH_CONV NUMSEG_CONV) THEN
       REWRITE_TAC[LOG_1; SUM_CLAUSES] THEN REAL_ARITH_TAC;
       ALL_TAC] THEN
     MP_TAC(ISPECL[`\z. clog(z)`; `\z. z * clog z - z`; `2`; `n:num`]
                  SUM_INTEGRAL_LBOUND_INCREASING) THEN
     REWRITE_TAC[] THEN ANTS_TAC THENL
      [CONJ_TAC THENL [POP_ASSUM MP_TAC THEN ARITH_TAC; ALL_TAC] THEN
       CONV_TAC "Examples/mangoldt.ml:REAL_RAT_REDUCE_CONV" REAL_RAT_REDUCE_CONV THEN CONJ_TAC THENL
        [REWRITE_TAC[IN_SEGMENT_CX_GEN] THEN REPEAT STRIP_TAC THENL
          [COMPLEX_DIFF_TAC THEN CONJ_TAC THEN UNDISCH_TAC `&1 <= Re x` THENL
            [REAL_ARITH_TAC; ALL_TAC] THEN
           ASM_CASES_TAC `x = Cx(&0)` THEN ASM_REWRITE_TAC[RE_CX] THENL
            [REAL_ARITH_TAC;
             UNDISCH_TAC `~(x = Cx(&0))` THEN CONV_TAC "Examples/mangoldt.ml:COMPLEX_FIELD" COMPLEX_FIELD];
           FIRST_X_ASSUM(MP_TAC o GEN_REWRITE_RULE I [GSYM REAL_OF_NUM_LE]) THEN
           ASM_REAL_ARITH_TAC];
        MAP_EVERY X_GEN_TAC [`a:real`; `b:real`] THEN STRIP_TAC THEN
        SUBGOAL_THEN `&0 < a /\ &0 < b` ASSUME_TAC THENL
         [ASM_REAL_ARITH_TAC; ALL_TAC] THEN
        ASM_SIMP_TAC[GSYM CX_LOG; RE_CX; LOG_MONO_LE_IMP]];
       ALL_TAC]] THEN
  CONV_TAC "Examples/mangoldt.ml:REAL_RAT_REDUCE_CONV" REAL_RAT_REDUCE_CONV THEN
  MATCH_MP_TAC(REAL_ARITH `y <= x /\ a <= b ==> x <= a ==> y <= b`) THEN
  ASM_SIMP_TAC[GSYM CX_LOG; SUM_EQ_NUMSEG; REAL_OF_NUM_LT; LE_1; CLOG_1;
               ARITH_RULE `2 <= n ==> 0 < n`; RE_CX;
               REAL_ARITH `&0 < &n + &1`; REAL_EQ_IMP_LE] THEN
  REWRITE_TAC[GSYM CX_MUL; GSYM CX_SUB; GSYM CX_ADD; RE_CX] THEN
  CONV_TAC "Examples/mangoldt.ml:REAL_RAT_REDUCE_CONV" REAL_RAT_REDUCE_CONV THEN REWRITE_TAC[REAL_SUB_RNEG] THENL
   [REWRITE_TAC[REAL_ARITH
     `(n + &1) * l' - (n + &1) + &1 <= (n * l - n + &1) + k * l <=>
      (n + &1) * l' <= (n + k) * l + &1`] THEN
    MATCH_MP_TAC REAL_LE_TRANS THEN
    EXISTS_TAC `(&n + &1) * (log(&n) + &1 / &n)` THEN CONJ_TAC THENL
     [MATCH_MP_TAC REAL_LE_LMUL THEN
      CONJ_TAC THENL [REAL_ARITH_TAC; ALL_TAC] THEN
      REWRITE_TAC[REAL_ARITH `x <= y + z <=> x - y <= z`] THEN
      ASM_SIMP_TAC[GSYM LOG_DIV; REAL_OF_NUM_LT; LT_NZ;
                   REAL_ARITH `&0 < &n + &1`;
                   REAL_FIELD `&0 < x ==> (x + &1) / x = &1 + &1 / x`] THEN
      MATCH_MP_TAC LOG_LE THEN SIMP_TAC[REAL_LE_DIV; REAL_POS];
      ALL_TAC] THEN
    REWRITE_TAC[REAL_ARITH
    `(n + &1) * (l + n') <= (n + k) * l + &1 <=>
      n' * (n + &1) <= (k - &1) * l + &1`] THEN
    ASM_SIMP_TAC[REAL_OF_NUM_EQ; REAL_LE_RADD; REAL_FIELD
     `~(n = &0) ==> &1 / n * (n + &1) = inv(n) + &1`] THEN
    MATCH_MP_TAC REAL_LE_TRANS THEN EXISTS_TAC `inv(&2)` THEN CONJ_TAC THENL
     [MATCH_MP_TAC REAL_LE_INV2 THEN
      REWRITE_TAC[REAL_OF_NUM_LE; REAL_OF_NUM_LT] THEN
      ASM_ARITH_TAC;
      ALL_TAC] THEN
    CONV_TAC "Examples/mangoldt.ml:REAL_RAT_REDUCE_CONV" REAL_RAT_REDUCE_CONV THEN REWRITE_TAC[REAL_MUL_LID] THEN
    MATCH_MP_TAC REAL_LE_TRANS THEN EXISTS_TAC `log(&2)` THEN
    REWRITE_TAC[LOG_2_BOUNDS] THEN MATCH_MP_TAC LOG_MONO_LE_IMP THEN
    REWRITE_TAC[REAL_OF_NUM_LE; REAL_OF_NUM_LE] THEN ASM_ARITH_TAC;
    SUBGOAL_THEN `&0 <= log(&n)` MP_TAC THENL [ALL_TAC; REAL_ARITH_TAC] THEN
    MATCH_MP_TAC LOG_POS THEN REWRITE_TAC[REAL_OF_NUM_LE] THEN
    ASM_ARITH_TAC]);;

(* ------------------------------------------------------------------------- *)
(* The Mangoldt function and its key expansion.                              *)
(* ------------------------------------------------------------------------- *)

let mangoldt = new_definition
 `mangoldt n = if ?p k. 1 <= k /\ prime p /\ n = p EXP k
               then log(&(@p. prime p /\ p divides n))
               else &0`;;

let MANGOLDT_1 = prove
 (`mangoldt 1 = &0`,
  REWRITE_TAC[mangoldt] THEN
  GEN_REWRITE_TAC "Examples/mangoldt.ml:(LAND_CONV o ONCE_DEPTH_CONV)" (LAND_CONV o ONCE_DEPTH_CONV) [EQ_SYM_EQ] THEN
  REWRITE_TAC[EXP_EQ_1] THEN MESON_TAC[PRIME_1; ARITH_RULE `~(1 <= 0)`]);;

let MANGOLDT_PRIMEPOW = prove
 (`!p k. prime p ==> mangoldt(p EXP k) = if 1 <= k then log(&p) else &0`,
  REPEAT STRIP_TAC THEN ASM_REWRITE_TAC[mangoldt] THEN
  ONCE_REWRITE_TAC[TAUT `a /\ b /\ c <=> ~(a /\ b ==> ~c)`] THEN
  ASM_SIMP_TAC[EQ_PRIME_EXP; LE_1] THEN
  REWRITE_TAC[TAUT `~(a /\ b ==> ~(c /\ d)) <=> d /\ c /\ a /\ b`] THEN
  ASM_REWRITE_TAC[UNWIND_THM1] THEN
  COND_CASES_TAC THEN ASM_REWRITE_TAC[] THEN REPEAT AP_TERM_TAC THEN
  ASM_SIMP_TAC[DIVIDES_PRIMEPOW] THEN MATCH_MP_TAC SELECT_UNIQUE THEN
  ASM_MESON_TAC[PRIME_DIVEXP; prime; PRIME_1; DIVIDES_REFL; EXP_1]);;

let MANGOLDT_POS_LE = prove
 (`!n. &0 <= mangoldt n`,
  GEN_TAC THEN ASM_CASES_TAC `?p k. 1 <= k /\ prime p /\ n = p EXP k` THENL
   [FIRST_X_ASSUM(REPEAT_TCL CHOOSE_THEN STRIP_ASSUME_TAC) THEN
    ASM_SIMP_TAC[MANGOLDT_PRIMEPOW] THEN MATCH_MP_TAC LOG_POS THEN
    REWRITE_TAC[REAL_OF_NUM_LE] THEN
    FIRST_X_ASSUM(MP_TAC o MATCH_MP PRIME_GE_2) THEN ARITH_TAC;
    ASM_REWRITE_TAC[mangoldt; REAL_LE_REFL]]);;

let LOG_MANGOLDT_SUM = prove
 (`!n. ~(n = 0) ==> log(&n) = sum {d | d divides n} (\d. mangoldt(d))`,
  REPEAT STRIP_TAC THEN ASM_CASES_TAC `n = 1` THENL
   [ASM_REWRITE_TAC[LOG_1; DIVIDES_ONE; SET_RULE `{x | x = a} = {a}`] THEN
    REWRITE_TAC[SUM_SING; mangoldt] THEN
    GEN_REWRITE_TAC "Examples/mangoldt.ml:(RAND_CONV o ONCE_DEPTH_CONV)" (RAND_CONV o ONCE_DEPTH_CONV) [EQ_SYM_EQ] THEN
    REWRITE_TAC[EXP_EQ_1] THEN MESON_TAC[PRIME_1; ARITH_RULE `~(1 <= 0)`];
    ALL_TAC] THEN
  SUBGOAL_THEN `1 < n` MP_TAC THENL
   [ASM_ARITH_TAC; ALL_TAC] THEN
  SPEC_TAC(`n:num`,`n:num`) THEN POP_ASSUM_LIST(K ALL_TAC) THEN
  MATCH_MP_TAC INDUCT_COPRIME THEN REPEAT STRIP_TAC THENL
   [ASM_SIMP_TAC[LOG_MUL; GSYM REAL_OF_NUM_MUL; REAL_OF_NUM_LT;
                 ARITH_RULE `1 < a ==> 0 < a`] THEN
    MATCH_MP_TAC EQ_TRANS THEN
    EXISTS_TAC
     `sum ({d | d divides a} UNION {d | d divides b}) (\d. mangoldt d)` THEN
    CONJ_TAC THEN CONV_TAC "Examples/mangoldt.ml:SYM_CONV" SYM_CONV THENL
     [MATCH_MP_TAC SUM_UNION_NONZERO THEN REWRITE_TAC[IN_INTER] THEN
      ASM_SIMP_TAC[FINITE_DIVISORS; ARITH_RULE `1 < n ==> ~(n = 0)`] THEN
      REWRITE_TAC[IN_ELIM_THM] THEN ASM_MESON_TAC[coprime; MANGOLDT_1];
      MATCH_MP_TAC SUM_SUPERSET THEN REWRITE_TAC[UNION_SUBSET; IN_UNION] THEN
      SIMP_TAC[SUBSET; IN_ELIM_THM; DIVIDES_LMUL; DIVIDES_RMUL] THEN
      X_GEN_TAC `d:num` THEN STRIP_TAC THEN REWRITE_TAC[mangoldt] THEN
      COND_CASES_TAC THEN ASM_REWRITE_TAC[] THEN
      ASM_MESON_TAC[PRIME_DIVPROD_POW]];
    ALL_TAC] THEN
  ASM_SIMP_TAC[DIVIDES_PRIMEPOW; GSYM REAL_OF_NUM_POW] THEN
  REWRITE_TAC[SET_RULE `{d | ?i. i <= k /\ d = p EXP i} =
                        IMAGE (\i. p EXP i) {i | i <= k}`] THEN
  ASM_SIMP_TAC[EQ_EXP; SUM_IMAGE; PRIME_GE_2;
               ARITH_RULE `2 <= p ==> ~(p = 0) /\ ~(p = 1)`] THEN
  ASM_SIMP_TAC[MANGOLDT_PRIMEPOW; o_DEF] THEN
  ASM_SIMP_TAC[GSYM SUM_RESTRICT_SET; IN_ELIM_THM; FINITE_NUMSEG_LE] THEN
  ONCE_REWRITE_TAC[CONJ_SYM] THEN REWRITE_TAC[GSYM numseg] THEN
  ASM_SIMP_TAC[LOG_POW; PRIME_IMP_NZ; REAL_OF_NUM_LT; LT_NZ] THEN
  SIMP_TAC[SUM_CONST; CARD_NUMSEG_1; FINITE_NUMSEG]);;

let MANGOLDT = prove
 (`!n. log(&(FACT n)) = sum(1..n) (\d. mangoldt(d) * floor(&n / &d))`,
  GEN_TAC THEN REWRITE_TAC[LOG_FACT] THEN MATCH_MP_TAC EQ_TRANS THEN
  EXISTS_TAC `sum(1..n) (\m. sum {d | d divides m} (\d. mangoldt d))` THEN
  SIMP_TAC[LOG_MANGOLDT_SUM; SUM_EQ_NUMSEG; LE_1] THEN
  MATCH_MP_TAC EQ_TRANS THEN
  EXISTS_TAC
   `sum (1..n) (\m. sum (1..n)
     (\d. mangoldt d * (if d divides m then &1 else &0)))` THEN
  CONJ_TAC THENL
   [MATCH_MP_TAC SUM_EQ_NUMSEG THEN X_GEN_TAC `m:num` THEN
    STRIP_TAC THEN REWRITE_TAC[] THEN CONV_TAC "Examples/mangoldt.ml:SYM_CONV" SYM_CONV THEN
    MATCH_MP_TAC SUM_EQ_SUPERSET THEN
    ASM_SIMP_TAC[LE_1; FINITE_DIVISORS; IN_ELIM_THM; REAL_MUL_RZERO;
                 REAL_MUL_RID; SUBSET; IN_NUMSEG] THEN
    GEN_TAC THEN DISCH_THEN(MP_TAC o MATCH_MP DIVIDES_LE_STRONG) THEN
    ASM_ARITH_TAC;
    GEN_REWRITE_TAC "Examples/mangoldt.ml:LAND_CONV" LAND_CONV [SUM_SWAP_NUMSEG] THEN
    MATCH_MP_TAC SUM_EQ_NUMSEG THEN X_GEN_TAC `d:num` THEN
    ASM_SIMP_TAC[SUM_DIVISORS_FLOOR_LEMMA; LE_1; SUM_LMUL]]);;

(* ------------------------------------------------------------------------- *)
(* The Chebyshev psi function and the key bounds on it.                      *)
(* ------------------------------------------------------------------------- *)

let PSI_BOUND_INDUCT = prove
 (`!n. ~(n = 0)
       ==> sum(1..2*n) (\d. mangoldt(d)) -
           sum(1..n) (\d. mangoldt(d)) <= &9 * &n`,
  REPEAT STRIP_TAC THEN MATCH_MP_TAC REAL_LE_TRANS THEN
  EXISTS_TAC `sum (n+1..2 * n) (\d. mangoldt d)` THEN CONJ_TAC THENL
   [MATCH_MP_TAC REAL_EQ_IMP_LE THEN REWRITE_TAC[REAL_EQ_SUB_RADD] THEN
    CONV_TAC "Examples/mangoldt.ml:SYM_CONV" SYM_CONV THEN MATCH_MP_TAC SUM_UNION_EQ THEN
    ONCE_REWRITE_TAC[UNION_COMM] THEN REWRITE_TAC[FINITE_NUMSEG] THEN
    ASM_SIMP_TAC[NUMSEG_COMBINE_R; ARITH_RULE
     `~(n = 0) ==> 1 <= n + 1 /\ n <= 2 * n`] THEN
    REWRITE_TAC[EXTENSION; IN_INTER; NOT_IN_EMPTY; IN_NUMSEG] THEN ARITH_TAC;
    ALL_TAC] THEN
  MATCH_MP_TAC REAL_LE_TRANS THEN EXISTS_TAC
   `sum (n+1..2*n)
        (\d. mangoldt(d) * (floor(&(2 * n) / &d) - &2 * floor(&n / &d)))` THEN
  CONJ_TAC THENL
   [MATCH_MP_TAC SUM_LE_NUMSEG THEN X_GEN_TAC `r:num` THEN STRIP_TAC THEN
    REWRITE_TAC[] THEN GEN_REWRITE_TAC "Examples/mangoldt.ml:LAND_CONV" LAND_CONV [GSYM REAL_MUL_RID] THEN
    MATCH_MP_TAC REAL_LE_LMUL THEN REWRITE_TAC[MANGOLDT_POS_LE] THEN
    MATCH_MP_TAC(REAL_ARITH `&1 <= a /\ b = &0 ==> &1 <= a - &2 * b`) THEN
    SUBGOAL_THEN `~(r = 0)` ASSUME_TAC THENL [ASM_ARITH_TAC; ALL_TAC] THEN
    ASM_SIMP_TAC[FLOOR_DIV_DIV; FLOOR_NUM; REAL_OF_NUM_LE; REAL_OF_NUM_EQ] THEN
    ASM_SIMP_TAC[DIV_EQ_0; LE_RDIV_EQ] THEN ASM_ARITH_TAC;
    ALL_TAC] THEN
  MATCH_MP_TAC REAL_LE_TRANS THEN EXISTS_TAC
   `sum (1..2*n)
        (\d. mangoldt(d) * (floor(&(2 * n) / &d) - &2 * floor(&n / &d)))` THEN
  CONJ_TAC THENL
   [MATCH_MP_TAC SUM_SUBSET THEN
    REWRITE_TAC[FINITE_NUMSEG; IN_DIFF; IN_NUMSEG] THEN
    CONJ_TAC THENL [ARITH_TAC; ALL_TAC] THEN
    X_GEN_TAC `r:num` THEN STRIP_TAC THEN
    SUBGOAL_THEN `~(r = 0)` ASSUME_TAC THENL [ASM_ARITH_TAC; ALL_TAC] THEN
    MATCH_MP_TAC REAL_LE_MUL THEN REWRITE_TAC[MANGOLDT_POS_LE] THEN
    ASM_SIMP_TAC[FLOOR_DIV_DIV; REAL_NEG_SUB; REAL_SUB_LE] THEN
    ASM_SIMP_TAC[REAL_OF_NUM_MUL; REAL_OF_NUM_LE; MULT_DIV_LE];
    ALL_TAC] THEN
  REWRITE_TAC[REAL_ARITH `m * (f1 - &2 * f2) = m * f1 - &2 * m * f2`] THEN
  REWRITE_TAC[SUM_SUB_NUMSEG; SUM_LMUL] THEN MATCH_MP_TAC REAL_LE_TRANS THEN
  EXISTS_TAC `sum(1..2*n) (\d. mangoldt(d) * floor(&(2 * n) / &d)) -
              &2 * sum(1..n) (\d. mangoldt(d) * floor(&n / &d))` THEN
  CONJ_TAC THENL
   [MATCH_MP_TAC(REAL_ARITH `y' <= y ==> x - y <= x - y'`) THEN
    MATCH_MP_TAC REAL_LE_LMUL THEN REWRITE_TAC[REAL_POS] THEN
    MATCH_MP_TAC SUM_SUBSET THEN
    REWRITE_TAC[FINITE_NUMSEG; IN_DIFF; IN_NUMSEG] THEN
    SIMP_TAC[FLOOR_DIV_DIV; LE_1; FLOOR_NUM; REAL_LE_MUL; REAL_POS;
             MANGOLDT_POS_LE] THEN
    ARITH_TAC;
    ALL_TAC] THEN
  REWRITE_TAC[GSYM MANGOLDT] THEN
  MAP_EVERY (MP_TAC o C SPEC LOG_FACT_BOUNDS) [`n:num`; `2 * n`] THEN
  ASM_REWRITE_TAC[MULT_EQ_0; ARITH_EQ] THEN
  MATCH_MP_TAC(REAL_ARITH
    `a2 + e2 + &2 * (e1 - a1) <= m
     ==> abs(f2 - a2) <= e2 ==> abs(f1 - a1) <= e1 ==> f2 - &2 * f1 <= m`) THEN
  ASM_SIMP_TAC[GSYM REAL_OF_NUM_MUL; LOG_MUL; REAL_OF_NUM_LT; LT_NZ; ARITH] THEN
  MATCH_MP_TAC REAL_LE_TRANS THEN
  EXISTS_TAC
   `&6 * log(&n) + (&2 * log(&2) - &1) * &1 + (&2 * log(&2)) * &n` THEN
  CONJ_TAC THENL [REAL_ARITH_TAC; ALL_TAC] THEN
  MATCH_MP_TAC REAL_LE_TRANS THEN
  EXISTS_TAC `&6 * &n + (&2 * log(&2) - &1) * &n + (&2 * log(&2)) * &n` THEN
  CONJ_TAC THENL
   [MATCH_MP_TAC REAL_LE_ADD2 THEN
    ASM_SIMP_TAC[LOG_LE_REFL; REAL_LE_LMUL; REAL_POS; REAL_LE_RADD] THEN
    MATCH_MP_TAC REAL_LE_LMUL THEN
    ASM_REWRITE_TAC[REAL_OF_NUM_LE; ARITH_RULE `1 <= n <=> ~(n = 0)`];
    REWRITE_TAC[GSYM REAL_ADD_RDISTRIB] THEN
    MATCH_MP_TAC REAL_LE_RMUL THEN REWRITE_TAC[REAL_POS]] THEN
  MP_TAC LOG_2_BOUNDS THEN REAL_ARITH_TAC);;

let PSI_BOUND_EXP = prove
 (`!n. sum(1..2 EXP n) (\d. mangoldt(d)) <= &9 * &(2 EXP n)`,
  INDUCT_TAC THEN
  SIMP_TAC[EXP; SUM_SING_NUMSEG; MANGOLDT_1; REAL_LE_MUL; REAL_POS] THEN
  REWRITE_TAC[GSYM REAL_OF_NUM_MUL] THEN
  FIRST_X_ASSUM(MATCH_MP_TAC o MATCH_MP (REAL_ARITH
   `s1 <= &9 * e ==> s2 - s1 <= &9 * e ==> s2 <= &9 * &2 * e`)) THEN
  MATCH_MP_TAC PSI_BOUND_INDUCT THEN
  REWRITE_TAC[EXP_EQ_0; ARITH]);;

let PSI_BOUND = prove
 (`!n. sum(1..n) (\d. mangoldt(d)) <= &18 * &n`,
  GEN_TAC THEN ASM_CASES_TAC `n <= 1` THENL
   [MATCH_MP_TAC REAL_LE_TRANS THEN
    EXISTS_TAC `sum(1..1) (\d. mangoldt d)` THEN CONJ_TAC THENL
     [MATCH_MP_TAC SUM_SUBSET; ALL_TAC] THEN
    REWRITE_TAC[SUM_SING_NUMSEG; FINITE_NUMSEG; IN_DIFF; IN_NUMSEG] THEN
    SIMP_TAC[MANGOLDT_POS_LE; MANGOLDT_1; REAL_LE_MUL; REAL_POS] THEN
    ASM_ARITH_TAC;
    ALL_TAC] THEN
  SUBGOAL_THEN `?k. n <= 2 EXP k /\ !l. l < k ==> ~(n <= 2 EXP l)`
  STRIP_ASSUME_TAC THENL
   [REWRITE_TAC[GSYM num_WOP] THEN EXISTS_TAC `n:num` THEN
    MP_TAC(SPEC `n:num` LT_POW2_REFL) THEN REWRITE_TAC[EXP] THEN ARITH_TAC;
    ALL_TAC] THEN
  MATCH_MP_TAC REAL_LE_TRANS THEN
  EXISTS_TAC `sum(1..2 EXP k) (\d. mangoldt d)` THEN CONJ_TAC THENL
   [MATCH_MP_TAC SUM_SUBSET THEN
    REWRITE_TAC[FINITE_NUMSEG; IN_DIFF; IN_NUMSEG; MANGOLDT_POS_LE] THEN
    ASM_ARITH_TAC;
    MATCH_MP_TAC REAL_LE_TRANS THEN EXISTS_TAC `&9 * &(2 EXP k)` THEN
    REWRITE_TAC[PSI_BOUND_EXP] THEN
    ASM_CASES_TAC `k = 0` THENL
     [FIRST_X_ASSUM SUBST_ALL_TAC THEN ASM_ARITH_TAC; ALL_TAC] THEN
    FIRST_ASSUM(SUBST1_TAC o MATCH_MP (ARITH_RULE
     `~(k = 0) ==> k = SUC(k - 1)`)) THEN
    FIRST_X_ASSUM(MP_TAC o SPEC `k - 1`) THEN
    ANTS_TAC THENL [ASM_ARITH_TAC; ALL_TAC] THEN
    REWRITE_TAC[REAL_OF_NUM_MUL; EXP; REAL_OF_NUM_LE] THEN ARITH_TAC]);;

(* ------------------------------------------------------------------------- *)
(* Now Mertens's first theorem.                                              *)
(* ------------------------------------------------------------------------- *)

let MERTENS_LEMMA = prove
 (`!n. ~(n = 0) ==> abs(sum(1..n) (\d. mangoldt(d) / &d) - log(&n)) <= &21`,
  REPEAT STRIP_TAC THEN MATCH_MP_TAC REAL_LE_LCANCEL_IMP THEN
  EXISTS_TAC `&n` THEN ASM_SIMP_TAC[REAL_OF_NUM_LT; LT_NZ] THEN
  GEN_REWRITE_TAC "Examples/mangoldt.ml:(LAND_CONV o LAND_CONV)" (LAND_CONV o LAND_CONV) [GSYM REAL_ABS_NUM] THEN
  REWRITE_TAC[GSYM REAL_ABS_MUL; REAL_SUB_LDISTRIB; GSYM SUM_LMUL] THEN
  FIRST_ASSUM(MP_TAC o MATCH_MP LOG_FACT_BOUNDS) THEN REWRITE_TAC[MANGOLDT] THEN
  MATCH_MP_TAC(REAL_ARITH
   `abs(n - &1) <= n /\ abs(s' - s) <= (k - &1) * n - a
    ==> abs(s' - (nl - n + &1)) <= a
        ==> abs(s - nl) <= n * k`) THEN
  CONJ_TAC THENL
   [MATCH_MP_TAC(REAL_ARITH `&1 <= x ==> abs(x - &1) <= x`) THEN
    REWRITE_TAC[REAL_OF_NUM_LE] THEN ASM_ARITH_TAC;
    ALL_TAC] THEN
  REWRITE_TAC[GSYM SUM_SUB_NUMSEG] THEN CONV_TAC "Examples/mangoldt.ml:REAL_RAT_REDUCE_CONV" REAL_RAT_REDUCE_CONV THEN
  ONCE_REWRITE_TAC[REAL_ARITH `n * i / x:real = i * n / x`] THEN
  REWRITE_TAC[GSYM REAL_SUB_LDISTRIB] THEN MATCH_MP_TAC REAL_LE_TRANS THEN
  EXISTS_TAC `sum(1..n) (\i. mangoldt i)` THEN CONJ_TAC THENL
   [MATCH_MP_TAC(REAL_ARITH `&0 <= --x /\ --x <= y ==> abs(x) <= y`) THEN
    REWRITE_TAC[GSYM SUM_NEG; REAL_ARITH
     `--(a * (x - y)):real = a * (y - x)`] THEN
    CONJ_TAC THENL
     [MATCH_MP_TAC SUM_POS_LE_NUMSEG THEN SIMP_TAC[] THEN
      X_GEN_TAC `i:num` THEN STRIP_TAC THEN MATCH_MP_TAC REAL_LE_MUL;
      MATCH_MP_TAC SUM_LE_NUMSEG THEN SIMP_TAC[] THEN
       X_GEN_TAC `i:num` THEN STRIP_TAC THEN
      GEN_REWRITE_TAC "Examples/mangoldt.ml:RAND_CONV" RAND_CONV [GSYM REAL_MUL_RID] THEN
      MATCH_MP_TAC REAL_LE_LMUL] THEN
    ASM_REWRITE_TAC[MANGOLDT_POS_LE; REAL_SUB_LE; REAL_LE_SUB_RADD] THEN
    MP_TAC(SPEC `&n / &i` FLOOR) THEN REAL_ARITH_TAC;
    ALL_TAC] THEN
  MATCH_MP_TAC(REAL_ARITH
   `x <= (k - &2) * n /\ l <= n ==> x <= k * n - &2 * l`) THEN
  ASM_SIMP_TAC[LOG_LE_REFL] THEN CONV_TAC "Examples/mangoldt.ml:REAL_RAT_REDUCE_CONV" REAL_RAT_REDUCE_CONV THEN
  ASM_REWRITE_TAC[PSI_BOUND]);;

let MERTENS_MANGOLDT_VERSUS_LOG = prove
 (`!n s. s SUBSET (1..n)
         ==> abs (sum s (\d. mangoldt d / &d) -
                  sum {p | prime p /\ p IN s} (\p. log (&p) / &p)) <= &3`,
  REPEAT GEN_TAC THEN ASM_CASES_TAC `n = 0` THENL
   [ASM_REWRITE_TAC[NUMSEG_CLAUSES; ARITH; SUBSET_EMPTY] THEN
    DISCH_THEN SUBST_ALL_TAC THEN
    REWRITE_TAC[NOT_IN_EMPTY; EMPTY_GSPEC; SUM_CLAUSES] THEN REAL_ARITH_TAC;
    DISCH_TAC] THEN
  MATCH_MP_TAC REAL_LE_TRANS THEN
  EXISTS_TAC `abs(sum (1..n) (\d. mangoldt d / &d) -
                  sum {p | prime p /\ p IN 1..n} (\p. log (&p) / &p))` THEN
  CONJ_TAC THENL
   [SUBGOAL_THEN `FINITE(s:num->bool)` ASSUME_TAC THENL
     [ASM_MESON_TAC[FINITE_SUBSET; FINITE_NUMSEG]; ALL_TAC] THEN
    ONCE_REWRITE_TAC[CONJ_SYM] THEN
    ASM_SIMP_TAC[SUM_RESTRICT_SET; FINITE_NUMSEG] THEN
    ASM_SIMP_TAC[GSYM SUM_SUB; FINITE_NUMSEG] THEN
    MATCH_MP_TAC(REAL_ARITH `&0 <= x /\ x <= y ==> abs x <= abs y`) THEN
    CONJ_TAC THENL
     [MATCH_MP_TAC SUM_POS_LE; MATCH_MP_TAC SUM_SUBSET_SIMPLE] THEN
    ASM_SIMP_TAC[IN_DIFF; FINITE_NUMSEG; REAL_SUB_LE] THEN
    X_GEN_TAC `x:num` THEN STRIP_TAC THEN COND_CASES_TAC THEN
    ASM_SIMP_TAC[REAL_LE_DIV; MANGOLDT_POS_LE; REAL_POS] THEN
    GEN_REWRITE_TAC "Examples/mangoldt.ml:(RAND_CONV o LAND_CONV o RAND_CONV)" (RAND_CONV o LAND_CONV o RAND_CONV) [GSYM EXP_1] THEN
    ASM_SIMP_TAC[MANGOLDT_PRIMEPOW; LE_REFL; REAL_LE_REFL];
    ALL_TAC] THEN
  SUBGOAL_THEN `{p | prime p /\ p IN 1..n} = {p | prime p /\ p <= n}`
  SUBST1_TAC THENL
   [REWRITE_TAC[EXTENSION; IN_ELIM_THM; IN_NUMSEG] THEN
    MESON_TAC[ARITH_RULE `2 <= p ==> 1 <= p`; PRIME_GE_2];
    ALL_TAC] THEN
  SUBGOAL_THEN
   `sum(1..n) (\d. mangoldt d / &d) -
    sum {p | prime p /\ p <= n} (\p. log (&p) / &p) =
    sum {p EXP k | prime p /\ p EXP k <= n /\ k >= 2} (\d. mangoldt d / &d)`
  SUBST1_TAC THENL
   [SUBGOAL_THEN
     `sum {p | prime p /\ p <= n} (\p. log (&p) / &p) =
      sum {p | prime p /\ p <= n} (\d. mangoldt d / &d)`
    SUBST1_TAC THENL
     [MATCH_MP_TAC SUM_EQ THEN REWRITE_TAC[IN_ELIM_THM] THEN
      REPEAT STRIP_TAC THEN
      GEN_REWRITE_TAC "Examples/mangoldt.ml:(RAND_CONV o LAND_CONV o RAND_CONV)" (RAND_CONV o LAND_CONV o RAND_CONV) [GSYM EXP_1] THEN
      ASM_SIMP_TAC[MANGOLDT_PRIMEPOW; ARITH];
      ALL_TAC] THEN
    REWRITE_TAC[REAL_EQ_SUB_RADD] THEN MATCH_MP_TAC EQ_TRANS THEN
    EXISTS_TAC
     `sum {p EXP k | prime p /\ p EXP k <= n /\ k >= 1}
          (\d. mangoldt d / &d)` THEN
    CONJ_TAC THENL
     [MATCH_MP_TAC SUM_SUPERSET THEN
      SIMP_TAC[IN_ELIM_THM; SUBSET; IN_NUMSEG] THEN
      CONJ_TAC THEN GEN_TAC THEN STRIP_TAC THEN
      ASM_REWRITE_TAC[ARITH_RULE `1 <= x <=> ~(x = 0)`; EXP_EQ_0] THENL
       [ASM_MESON_TAC[PRIME_0]; ALL_TAC] THEN
      REWRITE_TAC[real_div; REAL_ENTIRE] THEN DISJ1_TAC THEN
      REWRITE_TAC[mangoldt] THEN ASM_MESON_TAC[GE];
      ALL_TAC] THEN
    CONV_TAC "Examples/mangoldt.ml:SYM_CONV" SYM_CONV THEN MATCH_MP_TAC SUM_UNION_EQ THEN CONJ_TAC THENL
     [MATCH_MP_TAC FINITE_SUBSET THEN EXISTS_TAC `0..n` THEN
      REWRITE_TAC[SUBSET; IN_ELIM_THM; FINITE_NUMSEG; IN_NUMSEG; LE_0] THEN
      MESON_TAC[];
      ALL_TAC] THEN
    CONJ_TAC THENL
     [REWRITE_TAC[EXTENSION; IN_INTER; IN_ELIM_THM; NOT_IN_EMPTY] THEN
      MESON_TAC[PRIME_EXP; ARITH_RULE `~(1 >= 2)`];
      REWRITE_TAC[ARITH_RULE `k >= 1 <=> k >= 2 \/ k = 1`] THEN
      REWRITE_TAC[EXTENSION; IN_UNION; IN_ELIM_THM] THEN MESON_TAC[EXP_1]];
    ALL_TAC] THEN
  MATCH_MP_TAC(REAL_ARITH `&0 <= x /\ x <= y ==> abs(x) <= y`) THEN
  CONJ_TAC THENL
   [MATCH_MP_TAC SUM_POS_LE THEN
    SIMP_TAC[REAL_LE_DIV; REAL_POS; MANGOLDT_POS_LE]THEN
    MATCH_MP_TAC FINITE_SUBSET THEN EXISTS_TAC `0..n` THEN
    REWRITE_TAC[SUBSET; IN_ELIM_THM; FINITE_NUMSEG; IN_NUMSEG; LE_0] THEN
    MESON_TAC[];
    ALL_TAC] THEN
  MATCH_MP_TAC REAL_LE_TRANS THEN
  EXISTS_TAC
   `sum {p | p IN 1..n /\ prime p}
        (\p. sum (2..n) (\k. log(&p) / &p pow k))` THEN
  CONJ_TAC THENL
   [SIMP_TAC[SUM_SUM_PRODUCT; FINITE_NUMSEG; FINITE_RESTRICT] THEN
    MATCH_MP_TAC SUM_LE_INCLUDED THEN EXISTS_TAC `\(p,k). p EXP k` THEN
    SIMP_TAC[FINITE_PRODUCT; FINITE_NUMSEG; FINITE_RESTRICT] THEN
    CONJ_TAC THENL
     [MATCH_MP_TAC FINITE_SUBSET THEN EXISTS_TAC `0..n` THEN
      REWRITE_TAC[SUBSET; IN_ELIM_THM; FINITE_NUMSEG; IN_NUMSEG; LE_0] THEN
      MESON_TAC[];
      ALL_TAC] THEN
    REWRITE_TAC[FORALL_PAIR_THM; IN_ELIM_PAIR_THM; EXISTS_PAIR_THM] THEN
    SIMP_TAC[IN_ELIM_THM; IN_NUMSEG; REAL_LE_DIV; REAL_POW_LE; REAL_POS;
             LOG_POS; REAL_OF_NUM_LE] THEN
    X_GEN_TAC `x:num` THEN
    MATCH_MP_TAC MONO_EXISTS THEN X_GEN_TAC `p:num` THEN
    MATCH_MP_TAC MONO_EXISTS THEN X_GEN_TAC `k:num` THEN
    STRIP_TAC THEN FIRST_X_ASSUM SUBST_ALL_TAC THEN
    ASM_SIMP_TAC[MANGOLDT_PRIMEPOW; GSYM REAL_OF_NUM_POW; REAL_LE_REFL;
                 ARITH_RULE `k >= 2 ==> 1 <= k /\ 2 <= k`] THEN
    ASM_SIMP_TAC[PRIME_IMP_NZ; ARITH_RULE `1 <= k <=> ~(k = 0)`] THEN
    CONJ_TAC THENL
     [MATCH_MP_TAC LE_TRANS THEN EXISTS_TAC `p EXP k` THEN ASM_SIMP_TAC[] THEN
      GEN_REWRITE_TAC "Examples/mangoldt.ml:LAND_CONV" LAND_CONV [GSYM EXP_1] THEN
      ASM_SIMP_TAC[PRIME_IMP_NZ; LE_EXP] THEN ASM_ARITH_TAC;
      MATCH_MP_TAC LE_TRANS THEN EXISTS_TAC `p EXP k` THEN ASM_SIMP_TAC[] THEN
      MATCH_MP_TAC LE_TRANS THEN EXISTS_TAC `2 EXP k` THEN
      ASM_SIMP_TAC[LT_POW2_REFL; LT_IMP_LE; EXP_MONO_LE; PRIME_GE_2]];
    ALL_TAC] THEN
  REWRITE_TAC[real_div; SUM_LMUL; GSYM REAL_POW_INV; SUM_GP] THEN
  MATCH_MP_TAC REAL_LE_TRANS THEN
  EXISTS_TAC `sum {p | p IN 1..n /\ prime p}
                  (\p. log(&p) / (&p * (&p - &1)))` THEN
  CONJ_TAC THENL
   [MATCH_MP_TAC SUM_LE THEN SIMP_TAC[FINITE_NUMSEG; FINITE_RESTRICT] THEN
    X_GEN_TAC `p:num` THEN REWRITE_TAC[IN_ELIM_THM; IN_NUMSEG] THEN
    ASM_SIMP_TAC[REAL_INV_EQ_1; REAL_OF_NUM_EQ; PRIME_GE_2;
                 ARITH_RULE `2 <= p ==> ~(p = 1)`] THEN
    STRIP_TAC THEN COND_CASES_TAC THEN
    ASM_SIMP_TAC[REAL_MUL_RZERO; REAL_LE_DIV; REAL_LE_MUL; REAL_SUB_LE;
                 REAL_OF_NUM_LE; LOG_POS; LE_0] THEN
    REWRITE_TAC[real_div] THEN MATCH_MP_TAC REAL_LE_LMUL THEN
    ASM_SIMP_TAC[LOG_POS; REAL_OF_NUM_LE] THEN
    MATCH_MP_TAC(REAL_ARITH
     `&0 <= y * z /\ x * z <= a ==> (x - y) * z <= a`) THEN
    CONJ_TAC THENL
     [MATCH_MP_TAC REAL_LE_MUL THEN
      ASM_SIMP_TAC[REAL_POW_LE; REAL_LE_INV_EQ; REAL_POS; REAL_SUB_LE] THEN
      MATCH_MP_TAC REAL_INV_LE_1 THEN ASM_REWRITE_TAC[REAL_OF_NUM_LE];
      ALL_TAC] THEN
    MATCH_MP_TAC REAL_EQ_IMP_LE THEN
    FIRST_ASSUM(MP_TAC o MATCH_MP PRIME_GE_2) THEN
    REWRITE_TAC[GSYM REAL_OF_NUM_LE] THEN CONV_TAC "Examples/mangoldt.ml:REAL_FIELD" REAL_FIELD;
    ALL_TAC] THEN
  MATCH_MP_TAC REAL_LE_TRANS THEN
  EXISTS_TAC `sum (2..n)  (\p. log(&p) / (&p * (&p - &1)))` THEN
  CONJ_TAC THENL
   [MATCH_MP_TAC SUM_SUBSET THEN SIMP_TAC[FINITE_NUMSEG; FINITE_RESTRICT] THEN
    REWRITE_TAC[IN_DIFF; IN_NUMSEG; IN_ELIM_THM] THEN
    CONJ_TAC THENL [MESON_TAC[PRIME_GE_2]; ALL_TAC] THEN
    ASM_SIMP_TAC[LOG_POS; REAL_OF_NUM_LE; ARITH_RULE `2 <= p ==> 1 <= p`;
                 REAL_LE_MUL; REAL_POS; REAL_SUB_LE; REAL_LE_DIV];
    ALL_TAC] THEN
  MATCH_MP_TAC REAL_LE_TRANS THEN
  EXISTS_TAC `sum (2..n) (\m. log(&m) / (&m - &1) pow 2)` THEN CONJ_TAC THENL
   [MATCH_MP_TAC SUM_LE_NUMSEG THEN REPEAT STRIP_TAC THEN
    REWRITE_TAC[real_div] THEN MATCH_MP_TAC REAL_LE_LMUL THEN
    ASM_SIMP_TAC[LOG_POS; REAL_OF_NUM_LE; ARITH_RULE `2 <= p ==> 1 <= p`] THEN
    MATCH_MP_TAC REAL_LE_INV2 THEN
    ASM_SIMP_TAC[REAL_POW_2; REAL_LE_RMUL_EQ; REAL_LT_MUL; REAL_LT_IMP_LE;
                 REAL_SUB_LT; REAL_OF_NUM_LT; ARITH_RULE `1 < p <=> 2 <= p`;
                 REAL_ARITH `x - &1 <= x`];
    ALL_TAC] THEN
  ASM_CASES_TAC `n < 2` THENL
   [RULE_ASSUM_TAC(REWRITE_RULE[GSYM NUMSEG_EMPTY]);
    RULE_ASSUM_TAC(REWRITE_RULE[NOT_LT])] THEN
  ASM_SIMP_TAC[SUM_CLAUSES] THEN CONV_TAC "Examples/mangoldt.ml:REAL_RAT_REDUCE_CONV" REAL_RAT_REDUCE_CONV THEN
  ASM_SIMP_TAC[SUM_CLAUSES_LEFT; ARITH] THEN
  MATCH_MP_TAC(REAL_ARITH
    `x <= &1 /\ y <= e - &1 ==> x + y <= e`) THEN
  CONJ_TAC THENL [MP_TAC LOG_2_BOUNDS THEN REAL_ARITH_TAC; ALL_TAC] THEN
  ASM_CASES_TAC `n < 3` THENL
   [RULE_ASSUM_TAC(REWRITE_RULE[GSYM NUMSEG_EMPTY]);
    RULE_ASSUM_TAC(REWRITE_RULE[NOT_LT])] THEN
  ASM_SIMP_TAC[SUM_CLAUSES] THEN CONV_TAC "Examples/mangoldt.ml:REAL_RAT_REDUCE_CONV" REAL_RAT_REDUCE_CONV THEN
  MP_TAC(ISPECL
   [`\z. clog(z) / (z - Cx(&1)) pow 2`;
    `\z. clog(z - Cx(&1)) - clog(z) - clog(z) / (z - Cx(&1))`;
    `3`; `n:num`] SUM_INTEGRAL_UBOUND_DECREASING) THEN
  ASM_REWRITE_TAC[] THEN CONV_TAC "Examples/mangoldt.ml:REAL_RAT_REDUCE_CONV" REAL_RAT_REDUCE_CONV THEN ANTS_TAC THENL
   [CONJ_TAC THENL
     [REWRITE_TAC[IN_SEGMENT_CX_GEN] THEN X_GEN_TAC `z:complex` THEN
      STRIP_TAC THENL
       [COMPLEX_DIFF_TAC THEN SIMP_TAC[COMPLEX_SUB_RZERO; COMPLEX_MUL_LID] THEN
        ASM_SIMP_TAC[RE_SUB; RE_CX; REAL_SUB_LT] THEN
        ASM_SIMP_TAC[REAL_ARITH `&2 <= x ==> &1 < x /\ &0 < x`] THEN
        SUBGOAL_THEN `~(z = Cx(&0)) /\ ~(z = Cx(&1))` MP_TAC THENL
         [ALL_TAC; CONV_TAC "Examples/mangoldt.ml:COMPLEX_FIELD" COMPLEX_FIELD] THEN
        REPEAT STRIP_TAC THEN UNDISCH_TAC `&2 <= Re z` THEN
        ASM_REWRITE_TAC[RE_CX] THEN REAL_ARITH_TAC;
        RULE_ASSUM_TAC(REWRITE_RULE[GSYM REAL_OF_NUM_LE]) THEN
        ASM_ARITH_TAC];
      ALL_TAC] THEN
    MAP_EVERY X_GEN_TAC [`x:real`; `y:real`] THEN STRIP_TAC THEN
    MP_TAC(SPECL [`\z. clog(z) / (z - Cx(&1)) pow 2`;
                  `\z. inv(z * (z - Cx(&1)) pow 2) -
                       Cx(&2) * clog(z) / (z - Cx(&1)) pow 3`;
                  `Cx(x)`; `Cx(y)`] COMPLEX_MVT_LINE) THEN
    REWRITE_TAC[] THEN ANTS_TAC THENL
     [REWRITE_TAC[IN_SEGMENT_CX_GEN] THEN X_GEN_TAC `z:complex` THEN
      REWRITE_TAC[REAL_ARITH `a <= x /\ x <= b \/ b <= x /\ x <= a <=>
                         a <= x /\ x <= b \/ b < a /\ b <= x /\ x <= a`] THEN
      STRIP_TAC THENL [ALL_TAC; ASM_REAL_ARITH_TAC] THEN
      COMPLEX_DIFF_TAC THEN REWRITE_TAC[GSYM CONJ_ASSOC] THEN
      CONJ_TAC THENL [ASM_REAL_ARITH_TAC; ALL_TAC] THEN
      CONV_TAC "Examples/mangoldt.ml:NUM_REDUCE_CONV" NUM_REDUCE_CONV THEN
      SUBGOAL_THEN `~(z = Cx(&0)) /\ ~(z = Cx(&1))` MP_TAC THENL
       [ALL_TAC; CONV_TAC "Examples/mangoldt.ml:COMPLEX_FIELD" COMPLEX_FIELD] THEN
      CONJ_TAC THEN DISCH_THEN SUBST_ALL_TAC THEN
      REPEAT(POP_ASSUM MP_TAC) THEN REWRITE_TAC[RE_CX; IM_CX] THEN
      REAL_ARITH_TAC;
      ALL_TAC] THEN
    GEN_REWRITE_TAC "Examples/mangoldt.ml:RAND_CONV" RAND_CONV [REAL_ARITH `x <= y <=> x - y <= &0`] THEN
    DISCH_THEN(X_CHOOSE_THEN `w:complex`
     (CONJUNCTS_THEN2 ASSUME_TAC SUBST1_TAC)) THEN
    REWRITE_TAC[GSYM CX_SUB; RE_MUL_CX] THEN
    REWRITE_TAC[REAL_ARITH `a * (y - x) <= &0 <=> &0 <= --a * (y - x)`] THEN
    MATCH_MP_TAC REAL_LE_MUL THEN ASM_REWRITE_TAC[REAL_SUB_LE] THEN
    REWRITE_TAC[RE_SUB; REAL_NEG_SUB; REAL_SUB_LE] THEN
    SUBGOAL_THEN `real w` ASSUME_TAC THENL
     [ASM_MESON_TAC[REAL_SEGMENT; REAL_CX]; ALL_TAC] THEN
    FIRST_X_ASSUM(SUBST_ALL_TAC o SYM o GEN_REWRITE_RULE I [REAL]) THEN
    ABBREV_TAC `u = Re w` THEN
    FIRST_X_ASSUM(MP_TAC o GEN_REWRITE_RULE I [IN_SEGMENT_CX]) THEN
    ASM_SIMP_TAC[REAL_ARITH
     `x <= y
      ==> (x <= u /\ u <= y \/ y <= u /\ u <= x <=> x <= u /\ u <= y)`] THEN
    STRIP_TAC THEN
    SUBGOAL_THEN `&0 < u /\ &1 < u /\ &2 <= u` STRIP_ASSUME_TAC THENL
     [ASM_REAL_ARITH_TAC; ALL_TAC] THEN
    ASM_SIMP_TAC[GSYM CX_LOG; GSYM CX_SUB; GSYM CX_POW; GSYM CX_DIV;
                 GSYM CX_MUL; GSYM CX_INV; RE_CX] THEN
    REWRITE_TAC[REAL_POW_2; real_div; REAL_INV_MUL; REAL_MUL_ASSOC;
                REAL_RING `(x:real) pow 3 = x * x pow 2`] THEN
    ASM_SIMP_TAC[REAL_LE_RMUL_EQ; REAL_LT_INV_EQ; REAL_SUB_LT] THEN
    ASM_SIMP_TAC[GSYM real_div; REAL_LE_RDIV_EQ; REAL_SUB_LT] THEN
    MATCH_MP_TAC(REAL_ARITH
     `a * b <= &1 /\ &1 / &2 <= c ==> b * a <= &2 * c`) THEN
    ASM_SIMP_TAC[GSYM real_div; REAL_LE_LDIV_EQ] THEN
    CONJ_TAC THENL [ASM_REAL_ARITH_TAC; ALL_TAC] THEN
    MATCH_MP_TAC REAL_LE_TRANS THEN EXISTS_TAC `log(&2)` THEN
    REWRITE_TAC[LOG_2_BOUNDS] THEN MATCH_MP_TAC LOG_MONO_LE_IMP THEN
    ASM_REAL_ARITH_TAC;
    ALL_TAC] THEN
  CONV_TAC "Examples/mangoldt.ml:REAL_RAT_REDUCE_CONV" REAL_RAT_REDUCE_CONV THEN
  MATCH_MP_TAC(REAL_ARITH `x = y /\ a <= b ==> x <= a ==> y <= b`) THEN
  CONJ_TAC THENL [MATCH_MP_TAC SUM_EQ_NUMSEG; ALL_TAC] THEN
  ASM_SIMP_TAC[GSYM CX_SUB; GSYM CX_LOG; GSYM CX_DIV; REAL_SUB_LT; ARITH;
      RE_CX; REAL_OF_NUM_LT; ARITH_RULE `3 <= n ==> 0 < n /\ 1 < n`;
      GSYM CX_POW] THEN
  CONV_TAC "Examples/mangoldt.ml:REAL_RAT_REDUCE_CONV" REAL_RAT_REDUCE_CONV THEN
  REWRITE_TAC[LOG_1; REAL_ARITH `a - (&0 - x - x / &1) = a + &2 * x`] THEN
  MATCH_MP_TAC(REAL_ARITH
   `a <= e - &2 /\ x <= &1 ==> a + &2 * x <= e`) THEN
  REWRITE_TAC[LOG_2_BOUNDS] THEN
  MATCH_MP_TAC(REAL_ARITH `a <= b /\ --c <= e ==> a - b - c <= e`) THEN
  REWRITE_TAC[REAL_SUB_REFL; REAL_ARITH `--x <= &0 <=> &0 <= x`] THEN
  ASM_SIMP_TAC[REAL_LE_DIV; REAL_SUB_LE; LOG_POS; REAL_OF_NUM_LE;
    REAL_OF_NUM_LT; LOG_MONO_LE_IMP; REAL_ARITH `x - &1 <= x`; REAL_SUB_LT;
    LE_0; ARITH_RULE `3 <= n ==> 1 <= n /\ 1 < n`]);;

let MERTENS = prove
 (`!n. ~(n = 0)
       ==> abs(sum {p | prime p /\ p <= n}
                   (\p. log(&p) / &p) - log(&n)) <= &24`,
  REPEAT STRIP_TAC THEN
  FIRST_ASSUM(MP_TAC o MATCH_MP MERTENS_LEMMA) THEN
  MATCH_MP_TAC(REAL_ARITH
   `abs(s1 - s2) <= k - e ==> abs(s1 - l) <= e ==> abs(s2 - l) <= k`) THEN
  CONV_TAC "Examples/mangoldt.ml:REAL_RAT_REDUCE_CONV" REAL_RAT_REDUCE_CONV THEN
  SUBGOAL_THEN `{p | prime p /\ p <= n} = {p | prime p /\ p IN 1..n}`
  SUBST1_TAC THENL
   [REWRITE_TAC[EXTENSION; IN_ELIM_THM; IN_NUMSEG] THEN
    MESON_TAC[ARITH_RULE `2 <= p ==> 1 <= p`; PRIME_GE_2];
    MATCH_MP_TAC MERTENS_MANGOLDT_VERSUS_LOG THEN
    EXISTS_TAC `n:num` THEN ASM_REWRITE_TAC[SUBSET_REFL]]);;
