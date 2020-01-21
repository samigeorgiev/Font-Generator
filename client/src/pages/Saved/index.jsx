import React, {Component} from 'react';

import styles from './index.module.css';

class Saved extends Component {
    state = {
        loading: true,
        error: null,
        fonts: null
    };

    deleteHandler = async id => {
        const url = process.env.REACT_APP_BASE_URL + process.env.REACT_APP_DELETE_FONT_PATH + '/' + id;
        try {
            await fetch(url, {
                method: 'DELETE',
                headers: {
                    'Authorization': this.props.user.token
                }
            });
            this.setState(prevState => ({
                fonts: prevState.fonts.filter(font => font.id !== id)
            }));
        } catch (err) {
            this.setState({
                error: err
            });
        }
    };

    componentDidMount() {
        const url = process.env.REACT_APP_BASE_URL + process.env.REACT_APP_GET_SAVED_FONTS_PATH;
        fetch(url, {
            headers: {
                'Authorization': this.props.user.token
            }
        }).then(data => data.json()).then(fonts => {
            this.setState({
                fonts: fonts,
                loading: false
            });
        }).catch(err => this.setState({loading: false, error: err}));
    }

    render() {
        if (this.state.error) {
            throw this.state.error;
        }

        const pageContent = this.state.fonts && this.state.fonts.map(font => (
                <div key={font.id} className={styles.FontContainer}>
                    <p className={styles.HeadingFont}>Heading font: {font.heading}</p>
                    <p className={styles.BodyFont}>Body font: {font.body}</p>
                    <button className={styles.DeleteButton} onClick={() => this.deleteHandler(font.id)}>
                        X
                    </button>
                </div>
            ));

        return (
            <main className={styles.Saved}>
                {this.state.loading ? 'Loading..' : pageContent}
            </main>
        );
    }
}

export default Saved;