#pragma once

#include <array>
#include <cstddef>
#include <tuple>
namespace graphene::detail {
template <typename F, typename = void>  // used for SFINAE later
struct function_traits;

// specialization for function types
template <typename R, typename... A>
struct function_traits<R(A...)> {
  using return_type = R;
  using args_type = std::tuple<A...>;
  static constexpr std::size_t arity = sizeof...(A);
};

// specialization for function pointer types
template <typename R, typename... A>
struct function_traits<R (*)(A...)> {
  using return_type = R;
  using args_type = std::tuple<A...>;
  static constexpr std::size_t arity = sizeof...(A);
};

// specialization for pointers to member functions
template <typename F, typename>
struct function_traits : public function_traits<decltype(&F::operator())> {};
template <typename R, typename G, typename... A>
struct function_traits<R (G::*)(A...) const> {
  using return_type = R;
  using args_type = std::tuple<A...>;
  static constexpr std::size_t arity = sizeof...(A);
};

template <template <typename...> class T, typename Tuple>
struct apply_tuple_args;

template <template <typename...> class T, typename... Args>
struct apply_tuple_args<T, std::tuple<Args...>> {
  using type = T<Args...>;
};

template <typename ret_t, typename F, typename args_t, size_t... Is>
ret_t callFunctionWithUnpackedArgs(F& code, args_t args,
                                   std::index_sequence<Is...>) {
  if constexpr (std::is_same_v<ret_t, void>) {
    code(args[Is]...);
  } else {
    return code(args[Is]...);
  }
}

/**
 * Calls the given function with the unpacked arguments, i.e., the arguments in
 * the given vector are passed to the function as separate arguments.
 *
 * @tparam F The type of the function `code`.
 * @tparam ret_t The return type of the function `code`.
 * @param code The function to be called.
 * @param args The arguments to be unpacked and passed to the function `code`.
 * @return The result of the function `code` if `ret_t` is not `void`.
 */
template <typename ret_t, typename F, typename args_t>
ret_t callFunctionWithUnpackedArgs(F& code, args_t args) {
  if constexpr (std::is_same_v<ret_t, void>) {
    callFunctionWithUnpackedArgs<ret_t>(
        code, args, std::make_index_sequence<function_traits<F>::arity>{});
  } else
    return callFunctionWithUnpackedArgs<ret_t>(
        code, args, std::make_index_sequence<function_traits<F>::arity>{});
}

template <typename T, typename U>
concept invocable_with_args_of =
    std::invocable<T, U> ||                           // callable with one U
    std::invocable<T, U, U> ||                        // callable with two U's
    std::invocable<T, U, U, U> ||                     // callable with three U's
    std::invocable<T, U, U, U, U> ||                  // callable with four U's
    std::invocable<T, U, U, U, U, U> ||               // callable with five U's
    std::invocable<T, U, U, U, U, U, U> ||            // callable with six U's
    std::invocable<T, U, U, U, U, U, U, U> ||         // callable with seven U's
    std::invocable<T, U, U, U, U, U, U, U, U> ||      // callable with eight U's
    std::invocable<T, U, U, U, U, U, U, U, U, U> ||   // callable with nine U's
    std::invocable<T, U, U, U, U, U, U, U, U, U, U>;  // callable with ten U's

}  // namespace graphene::detail
