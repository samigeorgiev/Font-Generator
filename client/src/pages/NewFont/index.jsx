import React, {Component} from 'react';

import webFontLoader from 'webfontloader';

import Slider from 'components/Slider';
// import Spinner from 'components/Spinner';
import TextArea from 'components/TextArea';

import styles from './index.module.css';

class NewFont extends Component {
    state = {
        loading: true,
        error: null,
        properties: {
            contrast: {
                prev: 0,
                cur: 0
            },
            thickness: {
                prev: 0,
                cur: 0
            }
        },
        fonts: {
            heading: null,
            body: null
        },
        lastRequestTimeout: null
    };

    propertyHandler = async (property, e) => {
        clearTimeout(this.state.lastRequestTimeout);
        await this.setState({
            properties: {
                ...this.state.properties,
                [property]: {
                    ...this.state.properties[property],
                    cur: e.target.value
                }
            }
        });

        const url = process.env.REACT_APP_BASE_URL + process.env.REACT_APP_NEW_FONT_PATH;
        const options = {
            method: 'POST',
            headers: {
                'Authorization': this.props.user?.token,
                'Content-type': 'application/json'
            },
            body: JSON.stringify({
                deltaContrast: this.state.properties.contrast.cur - this.state.properties.contrast.prev,
                deltaThickness: this.state.properties.thickness.cur - this.state.properties.thickness.prev,
                fonts: this.state.fonts
            })
        };
        const timeout = setTimeout(() => fetch(url, options).then(data => data.json()).then(fonts => {
            this.changeFonts({
                heading: fonts.heading,
                body: fonts.body
            });
        }).catch(err => {
            console.log(err);
        }), 1000);
        this.setState({
            lastRequestTimeout: timeout
        });
    };

    saveHandler = async () => {
        const url = process.env.REACT_APP_BASE_URL + process.env.REACT_APP_SAVE_FONT_PATH;
        const options = {
            method: 'POST',
            headers: {
                'Authorization': this.props.user.token,
                'Content-type': 'application/json'
            },
            body: JSON.stringify({fonts: this.state.fonts})
        };
        try {
            await fetch(url, options);
        } catch (err) {
            this.setState({
                error: err
            });
        }
    };

    changeFonts = async ({heading, body}) => {
        webFontLoader.load({
            google: {
                families: [heading, body]
            },
        });
        this.setState({
            fonts: {
                heading: heading,
                body: body
            }
        });
    };

    componentDidMount() {
        const url = process.env.REACT_APP_BASE_URL + process.env.REACT_APP_RECOMMEND_PATH;
        fetch(url, {
            headers: {
                'Authorization': this.props.user?.token
            }
        }).then(data => data.json()).then(fonts => {
            this.setState({
                fonts: {
                    heading: fonts.heading,
                    body: fonts.body
                },
                loading: false
            });
        }).catch(err => this.setState({ error: err }));
    }

    render() {
        if (this.state.error) { throw this.state.error; }

        const pageContent = (
            <>
                <div className={styles.Headings}>
                    <h2>Heading font: {this.state.fonts.heading}</h2><h2>Body font: {this.state.fonts.body}</h2>
                </div>
                <div className={styles.FirstRowContainer}>
                    {Object.entries(this.state.properties).map(([property, value]) => (
                        <Slider
                            key={property}
                            min="-1"
                            max="1"
                            step="0.01"
                            name={property}
                            value={value.cur}
                            change={e => this.propertyHandler(property, e)}
                        />
                    ))}
                    {this.props.user
                        ? <button className={styles.SaveButton} onClick={this.saveHandler}>
                            SAVE
                        </button>
                        : null}
                </div>
                <TextArea font={this.state.fonts.heading} size="5rem">
                    Heading 1
                </TextArea>
                <TextArea font={this.state.fonts.heading} size="3rem">
                    Heading 2
                </TextArea>
                <TextArea font={this.state.fonts.body} size="1rem">
                    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus ut suscipit dolor, vitae fermentum
                    dui. Suspendisse potenti. Quisque eleifend malesuada nisi vitae molestie. Donec aliquam purus non
                    diam elementum, ac faucibus justo fringilla. Ut lobortis porta velit vel gravida. Aliquam eget purus
                    ac nibh euismod rutrum. Pellentesque non elit sed.
                </TextArea>
            </>
        );

        return (
            <main className={styles.NewFont}>
                {this.state.loading ? 'Loading...' : pageContent}
            </main>
        );
    }
}

export default NewFont;