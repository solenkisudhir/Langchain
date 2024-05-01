# Streaming | 🦜️🔗 LangChain
All `LLM`s implement the `Runnable` interface, which comes with default implementations of all methods, ie. ainvoke, batch, abatch, stream, astream. This gives all `LLM`s basic support for streaming.

Streaming support defaults to returning an Iterator (or AsyncIterator in the case of async streaming) of a single value, the final result returned by the underlying `LLM` provider. This obviously doesn’t give you token-by-token streaming, which requires native support from the `LLM` provider, but ensures your code that expects an iterator of tokens can work for any of our `LLM` integrations.

```


Verse 1:
Bubbles dancing in my glass
Clear and crisp, it's such a blast
Refreshing taste, it's like a dream
Sparkling water, you make me beam

Chorus:
Oh sparkling water, you're my delight
With every sip, you make me feel so right
You're like a party in my mouth
I can't get enough, I'm hooked no doubt

Verse 2:
No sugar, no calories, just pure bliss
You're the perfect drink, I must confess
From lemon to lime, so many flavors to choose
Sparkling water, you never fail to amuse

Chorus:
Oh sparkling water, you're my delight
With every sip, you make me feel so right
You're like a party in my mouth
I can't get enough, I'm hooked no doubt

Bridge:
Some may say you're just plain water
But to me, you're so much more
You bring a sparkle to my day
In every single way

Chorus:
Oh sparkling water, you're my delight
With every sip, you make me feel so right
You're like a party in my mouth
I can't get enough, I'm hooked no doubt

Outro:
So here's to you, my dear sparkling water
You'll always be my go-to drink forever
With your effervescence and refreshing taste
You'll always have a special place.

```
